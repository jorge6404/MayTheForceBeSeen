cmake_minimum_required(VERSION 3.16)
project(visual_normal_force CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ONNX Runtime: release precompilado (onnxruntime-linux-x64[-gpu]-X.Y.Z.tgz) o el de JetPack.
# -DONNXRUNTIME_ROOT=/ruta o variable de entorno ONNXRUNTIME_ROOT; si no, se busca en /usr y /usr/local.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Raíz de ONNX Runtime (include/ y lib/)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime no encontrado: define ONNXRUNTIME_ROOT (carpeta con include/ y lib/)")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
# Que los ejecutables instalados encuentren libonnxruntime sin LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Solo la usan el nodo y los tests: estática y sin instalar cabeceras
add_library(${PROJECT_NAME}_core STATIC src/inference.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_core PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

add_executable(normal_force_node src/normal_force_node.cpp)
target_link_libraries(normal_force_node ${PROJECT_NAME}_core)
ament_target_dependencies(normal_force_node rclcpp sensor_msgs std_msgs ament_index_cpp)

# Modelo: Hugging Face, fijado a un commit y comprobado por SHA-256 (no está en git: pesa 106 MB).
# Sin internet: copia el .onnx en models/ antes de compilar y se usa ese si el hash coincide.
set(MODEL_URL "https://huggingface.co/jorge6404/visual-normal-force/resolve/9bc3a7f488fe447fb91a793ffadcdfde4fdbbdd5/normal_force.onnx")
set(MODEL_SHA256 "782921832cbb2bbb4a1154fb51c2faff677aa398f4d4fac8924c92e7c1cbf265")
set(MODEL_FILE "${CMAKE_CURRENT_BINARY_DIR}/models/normal_force.onnx")
set(MODEL_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/models/normal_force.onnx")

function(model_ok path out)
  set(${out} FALSE PARENT_SCOPE)
  if(EXISTS "${path}")
    file(SHA256 "${path}" hash)
    if(hash STREQUAL MODEL_SHA256)
      set(${out} TRUE PARENT_SCOPE)
    endif()
  endif()
endfunction()

model_ok("${MODEL_FILE}" ya_esta)
model_ok("${MODEL_LOCAL}" hay_local)
if(NOT ya_esta AND hay_local)
  file(COPY "${MODEL_LOCAL}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/models")
elseif(NOT ya_esta)
  message(STATUS "Descargando el modelo (106 MB): ${MODEL_URL}")
  file(DOWNLOAD "${MODEL_URL}" "${MODEL_FILE}" EXPECTED_HASH SHA256=${MODEL_SHA256} SHOW_PROGRESS STATUS st)
  list(GET st 0 code)
  if(NOT code EQUAL 0)
    file(REMOVE "${MODEL_FILE}")
    message(FATAL_ERROR "No se pudo descargar el modelo (${st}). Sin internet: copia normal_force.onnx en ${MODEL_LOCAL}")
  endif()
endif()

install(TARGETS normal_force_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES "${MODEL_FILE}" DESTINATION share/${PROJECT_NAME}/models)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_inference test/test_inference.cpp TIMEOUT 120)
  target_link_libraries(test_inference ${PROJECT_NAME}_core)
  target_compile_definitions(test_inference PRIVATE
    MODEL_PATH="${MODEL_FILE}"
    TEST_IMAGE="${CMAKE_CURRENT_SOURCE_DIR}/test/data/rosbag_fin_5_000000.jpg")

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_node.launch.py TIMEOUT 120)
endif()

ament_package()
