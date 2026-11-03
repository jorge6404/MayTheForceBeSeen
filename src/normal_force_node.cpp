// Nodo: sensor_msgs/CompressedImage → fuerza normal de la red sobre el ROV (N, std_msgs/Float64).
//
// La suscripción solo guarda la última imagen. Un temporizador a rate_hz, en su propio grupo de callbacks (otro hilo
// del MultiThreadedExecutor), la toma, infiere y publica, así las imágenes siguen llegando mientras se infiere.
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float64.hpp>

#include "visual_normal_force/inference.hpp"

namespace visual_normal_force
{

class NormalForceNode : public rclcpp::Node
{
public:
  explicit NormalForceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("normal_force", options),
    engine_(declare_parameter<std::string>("engine", "auto")),
    model_(ament_index_cpp::get_package_share_directory("visual_normal_force") + "/models/normal_force.onnx",
      engine_)
  {
    const auto in_topic = declare_parameter<std::string>(
      "input_topic", "/cirtesub/stereo_left_camera/camera/image_raw/compressed");
    const auto out_topic = declare_parameter<std::string>("output_topic", "/visual_normal_force/force");
    const double rate_hz = declare_parameter<double>("rate_hz", 5.0);
    if (rate_hz <= 0.0) {
      throw std::invalid_argument("rate_hz debe ser > 0");
    }
    log_engine();

    // Cámara: best effort y solo la última (compatible con publishers reliable y best effort)
    sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      in_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lk(mutex_);
        last_ = std::move(msg);
      });
    pub_ = create_publisher<std_msgs::msg::Float64>(out_topic, 10);
    // El nodo solo guarda weak_ptr de sus grupos: si el grupo no es un miembro, se destruye y el timer no salta nunca
    inference_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz), [this] {on_timer();}, inference_group_);
    RCLCPP_INFO(get_logger(), "%s → %s a %g Hz", in_topic.c_str(), out_topic.c_str(), rate_hz);
  }

private:
  void log_engine()
  {
    std::string why;
    for (const auto & f : model_.failed()) {
      why += "\n  " + f.engine + ": " + f.reason;
    }
    const std::string used(model_.engine());
    if (engine_ != "auto" && engine_ != used) {
      RCLCPP_WARN(get_logger(), "Motor pedido '%s' no disponible, se usa '%s':%s",
                  engine_.c_str(), used.c_str(), why.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Motor: %s (pedido: %s)%s", used.c_str(), engine_.c_str(), why.c_str());
    }
  }

  void on_timer()
  {
    sensor_msgs::msg::CompressedImage::ConstSharedPtr msg;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      msg = std::exchange(last_, nullptr);  // nula si no ha llegado otra desde la última inferencia
    }
    if (!msg) {
      return;
    }
    try {
      const auto x = preprocess(msg->data.data(), msg->data.size(), model_.input_size());
      if (x.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No se pudo decodificar la imagen (%s)",
                             msg->format.c_str());
        return;
      }
      std_msgs::msg::Float64 out;
      out.data = model_(x);
      pub_->publish(out);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Inferencia fallida: %s", e.what());
    }
  }

  const std::string engine_;
  NormalForceModel model_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_;
  rclcpp::CallbackGroup::SharedPtr inference_group_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex mutex_;
  sensor_msgs::msg::CompressedImage::ConstSharedPtr last_;
};

}  // namespace visual_normal_force

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  auto node = std::make_shared<visual_normal_force::NormalForceNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
