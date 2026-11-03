#include "visual_normal_force/inference.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace visual_normal_force
{

namespace
{

void append_cuda(Ort::SessionOptions & so)
{
  const OrtApi & api = Ort::GetApi();
  OrtCUDAProviderOptionsV2 * cuda = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> guard(
    cuda, api.ReleaseCUDAProviderOptions);
  so.AppendExecutionProvider_CUDA_V2(*cuda);
}

void append_tensorrt(Ort::SessionOptions & so)
{
  // Caché de engines: compilar uno tarda minutos, solo la primera vez
  const char * home = std::getenv("HOME");
  const std::string cache = std::string(home ? home : "/tmp") + "/.cache/visual_normal_force";
  std::filesystem::create_directories(cache);

  const OrtApi & api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2 * trt = nullptr;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> guard(
    trt, api.ReleaseTensorRTProviderOptions);
  const char * keys[] = {"trt_engine_cache_enable", "trt_engine_cache_path", "trt_fp16_enable"};
  const char * values[] = {"1", cache.c_str(), "1"};
  Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys, values, std::size(keys)));
  so.AppendExecutionProvider_TensorRT_V2(*trt);
  append_cuda(so);  // para los nodos que TensorRT no soporte
}

void append_cpu(Ort::SessionOptions &) {}

// En orden de preferencia: pedir un motor es pedir ese o, si no arranca, los siguientes.
const Backend kBackends[] = {
  {"tensorrt", "TensorrtExecutionProvider", append_tensorrt},
  {"cuda", "CUDAExecutionProvider", append_cuda},
  {"cpu", "CPUExecutionProvider", append_cpu},
};

Ort::Session open_session(Ort::Env & env, const std::string & path, const Backend & backend)
{
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  backend.append(so);
  return Ort::Session(env, path.c_str(), so);
}

NormalForceModel::Io describe(const Ort::Session & s)
{
  const auto shape = s.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0) {
    throw std::runtime_error("el modelo debe tener una entrada fija 1×3×alto×ancho");
  }
  Ort::AllocatorWithDefaultOptions alloc;
  return {s.GetInputNameAllocated(0, alloc).get(), s.GetOutputNameAllocated(0, alloc).get(),
    cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]))};
}

float run(Ort::Session & s, const NormalForceModel::Io & io, const std::vector<float> & input)
{
  const std::array<int64_t, 4> shape{1, 3, io.size.height, io.size.width};
  const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // ORT no modifica la entrada; la API solo acepta float*
  auto tensor = Ort::Value::CreateTensor<float>(
    mem, const_cast<float *>(input.data()), input.size(), shape.data(), shape.size());
  const char * in[] = {io.input.c_str()};
  const char * out[] = {io.output.c_str()};
  auto result = s.Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);
  return result.front().GetTensorData<float>()[0];
}

}  // namespace

std::vector<float> preprocess(const uint8_t * data, size_t size, cv::Size input_size)
{
  if (data == nullptr || size == 0) {
    return {};
  }
  const cv::Mat bgr = cv::imdecode(cv::Mat(1, static_cast<int>(size), CV_8UC1, const_cast<uint8_t *>(data)),
                                   cv::IMREAD_COLOR);
  if (bgr.empty()) {
    return {};
  }
  cv::Mat small, scaled;
  cv::resize(bgr, small, input_size, 0, 0, cv::INTER_AREA);
  small.convertTo(scaled, CV_32F, 1.0 / 255.0);

  // split escribe cada canal directamente en su plano del tensor: B → plano 2, G → 1, R → 0 (queda RGB)
  const size_t area = input_size.area();
  std::vector<float> out(3 * area);
  std::vector<cv::Mat> planes{cv::Mat(input_size, CV_32F, out.data() + 2 * area),
    cv::Mat(input_size, CV_32F, out.data() + area), cv::Mat(input_size, CV_32F, out.data())};
  cv::split(scaled, planes);
  return out;
}

std::vector<const Backend *> candidates(std::string_view engine)
{
  const auto first = engine == "auto" ? std::begin(kBackends) :
    std::find_if(std::begin(kBackends), std::end(kBackends), [&](const Backend & b) {return b.name == engine;});
  if (first == std::end(kBackends)) {
    throw std::invalid_argument("engine '" + std::string(engine) + "' no válido (auto|tensorrt|cuda|cpu)");
  }
  std::vector<const Backend *> out;
  for (auto it = first; it != std::end(kBackends); ++it) {
    out.push_back(&*it);
  }
  return out;
}

NormalForceModel::NormalForceModel(const std::string & model_path, std::string_view engine)
: env_(ORT_LOGGING_LEVEL_WARNING, "visual_normal_force")
{
  const auto available = Ort::GetAvailableProviders();
  for (const Backend * b : candidates(engine)) {
    if (std::find(available.begin(), available.end(), b->ep) == available.end()) {
      failed_.push_back({std::string(b->name), "no incluido en este ONNX Runtime"});
      continue;
    }
    // Cada motor en su propia sesión: si un provider listado no carga (p. ej. falta libnvinfer), no arrastra a los
    // demás. La inferencia de calentamiento construye el engine de TensorRT / reserva la memoria en GPU.
    try {
      Ort::Session session = open_session(env_, model_path, *b);
      Io io = describe(session);
      run(session, io, std::vector<float>(3 * io.size.area(), 0.5f));
      session_ = std::move(session);
      io_ = std::move(io);
      engine_ = b->name;
      return;
    } catch (const std::exception & e) {
      failed_.push_back({std::string(b->name), e.what()});
    }
  }
  throw std::runtime_error("no se pudo crear la sesión ONNX con ningún motor: " + failed_.back().reason);
}

float NormalForceModel::operator()(const std::vector<float> & input)
{
  return run(session_, io_, input);
}

}  // namespace visual_normal_force
