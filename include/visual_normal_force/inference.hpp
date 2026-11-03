// Preprocesado y sesión ONNX Runtime para el modelo imagen → fuerza normal. Sin ROS, para poder probarlo aparte.
//
// El grafo ONNX ya normaliza con ImageNet y devuelve newtons (y·sigma + mu), así que aquí solo hay que decodificar,
// reescalar al tamaño de entrada del modelo con INTER_AREA (como en el entrenamiento), pasar a RGB y dividir por 255.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

namespace visual_normal_force
{

// Bytes de una imagen comprimida (JPEG/PNG) → tensor float32 1×3×alto×ancho, RGB en [0, 1], NCHW.
// Vacío si no se puede decodificar.
std::vector<float> preprocess(const uint8_t * data, size_t size, cv::Size input_size);

// Motor de inferencia: nombre del parámetro `engine`, execution provider de ORT y cómo añadirlo a la sesión.
struct Backend
{
  std::string_view name;
  const char * ep;
  void (*append)(Ort::SessionOptions &);
};

// Motores a intentar, de mejor a peor: el pedido (auto|tensorrt|cuda|cpu) y los que le siguen en preferencia.
// Lanza std::invalid_argument si el pedido no es válido.
std::vector<const Backend *> candidates(std::string_view engine);

struct Failure
{
  std::string engine;
  std::string reason;
};

class NormalForceModel
{
public:
  // Prueba los motores de candidates() uno a uno (sesión + inferencia de calentamiento) y se queda con el primero
  // que funciona. Lanza std::runtime_error si no arranca ninguno.
  NormalForceModel(const std::string & model_path, std::string_view engine);

  // Fuerza normal en N para un tensor de preprocess() con input_size().
  float operator()(const std::vector<float> & input);

  cv::Size input_size() const {return io_.size;}
  std::string_view engine() const {return engine_;}
  // Motores que se intentaron (o se pidieron) y no se usan, con el motivo.
  const std::vector<Failure> & failed() const {return failed_;}

  struct Io
  {
    std::string input, output;
    cv::Size size;
  };

private:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Io io_;
  std::string_view engine_;
  std::vector<Failure> failed_;
};

}  // namespace visual_normal_force
