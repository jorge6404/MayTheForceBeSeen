#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "visual_normal_force/inference.hpp"

using visual_normal_force::candidates;
using visual_normal_force::NormalForceModel;
using visual_normal_force::preprocess;

namespace
{
const cv::Size kSize(384, 288);

std::vector<uint8_t> encode(const cv::Mat & img, const std::string & ext)
{
  std::vector<uint8_t> buf;
  cv::imencode(ext, img, buf);
  return buf;
}

std::vector<std::string_view> names(std::string_view engine)
{
  std::vector<std::string_view> out;
  for (const auto * b : candidates(engine)) {
    out.push_back(b->name);
  }
  return out;
}
}  // namespace

TEST(Preprocess, ShapeAndRange)
{
  cv::Mat img(480, 640, CV_8UC3);
  cv::randu(img, 0, 255);
  const auto buf = encode(img, ".jpg");
  const auto x = preprocess(buf.data(), buf.size(), kSize);
  ASSERT_EQ(x.size(), 3u * kSize.area());
  for (float v : x) {
    ASSERT_GE(v, 0.0f);
    ASSERT_LE(v, 1.0f);
  }
}

TEST(Preprocess, RgbOrder)
{
  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(10, 128, 255));  // B, G, R
  const auto buf = encode(img, ".png");
  const auto x = preprocess(buf.data(), buf.size(), kSize);
  const size_t area = kSize.area();
  EXPECT_FLOAT_EQ(x[0], 255 / 255.0f);          // plano 0 = R
  EXPECT_FLOAT_EQ(x[area], 128 / 255.0f);       // plano 1 = G
  EXPECT_FLOAT_EQ(x[2 * area], 10 / 255.0f);    // plano 2 = B
  EXPECT_FLOAT_EQ(x[3 * area - 1], 10 / 255.0f);
}

TEST(Preprocess, Invalid)
{
  const std::vector<uint8_t> junk{1, 2, 3, 4};
  EXPECT_TRUE(preprocess(junk.data(), junk.size(), kSize).empty());
  EXPECT_TRUE(preprocess(nullptr, 0, kSize).empty());
}

TEST(Candidates, Order)
{
  using V = std::vector<std::string_view>;
  EXPECT_EQ(names("auto"), (V{"tensorrt", "cuda", "cpu"}));
  EXPECT_EQ(names("tensorrt"), (V{"tensorrt", "cuda", "cpu"}));
  EXPECT_EQ(names("cuda"), (V{"cuda", "cpu"}));
  EXPECT_EQ(names("cpu"), (V{"cpu"}));
  EXPECT_THROW(candidates("xx"), std::invalid_argument);
}

// Imagen real de la rosbag de test (rosbag_fin_5/000000.jpg): el .pt de la semilla 2 predice −7,53 N y el ONNX en
// Python −7,51 N. Comprueba que preprocesado + modelo en C++ dan lo mismo y que el tamaño sale del modelo.
TEST(Model, RealImageCpu)
{
  std::ifstream f(TEST_IMAGE, std::ios::binary);
  ASSERT_TRUE(f.good()) << TEST_IMAGE;
  const std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  NormalForceModel model(MODEL_PATH, "cpu");
  EXPECT_EQ(model.engine(), "cpu");
  EXPECT_EQ(model.input_size(), kSize);
  EXPECT_NEAR(model(preprocess(buf.data(), buf.size(), model.input_size())), -7.51f, 0.1f);
}

// Pedir un motor que este ONNX Runtime no trae debe quedar registrado, no descartarse en silencio.
TEST(Model, RequestedEngineUnavailableIsReported)
{
  const auto available = Ort::GetAvailableProviders();
  if (std::find(available.begin(), available.end(), "CUDAExecutionProvider") != available.end()) {
    GTEST_SKIP() << "este ONNX Runtime trae CUDA";
  }
  NormalForceModel model(MODEL_PATH, "cuda");
  EXPECT_EQ(model.engine(), "cpu");
  ASSERT_EQ(model.failed().size(), 1u);
  EXPECT_EQ(model.failed()[0].engine, "cuda");
}
