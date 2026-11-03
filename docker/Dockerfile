# ROS 2 Humble + ONNX Runtime C++ (release precompilado) + visual_normal_force. Contexto de build: la carpeta del
# paquete. Una sola receta para las dos variantes (docker-compose.yml pasa los argumentos):
#   CPU:    BASE=ubuntu:22.04                                  ORT_FLAVOR=
#   NVIDIA: BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04  ORT_FLAVOR=-gpu   (CUDA EP; sin TensorRT)
ARG BASE=ubuntu:22.04
FROM ${BASE}
ARG ORT_FLAVOR=
ARG ORT_VERSION=1.23.2
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base build-essential cmake libopencv-dev python3-colcon-common-extensions \
        ros-humble-ament-cmake-gtest ros-humble-launch-testing-ament-cmake \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/onnxruntime && curl -fsSL --retry 5 --retry-all-errors \
        https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64${ORT_FLAVOR}-${ORT_VERSION}.tgz \
    | tar -xz -C /opt/onnxruntime --strip-components=1
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

COPY . /ws/src/visual_normal_force
WORKDIR /ws
RUN . /opt/ros/humble/setup.sh && colcon build --packages-select visual_normal_force \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "visual_normal_force", "normal_force.launch.py", "params_file:=/config/normal_force.yaml"]
