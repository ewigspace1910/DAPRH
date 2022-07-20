cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(sample_onnx LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
# cuda compilier
enable_language(CUDA)
find_package(OpenCV REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(onnx_trt SHARED
  ./engine.h
  ./engine.cpp
)
set_target_properties(onnx_trt PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_trt PUBLIC nvinfer nvonnxparser ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})

add_executable(export export.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(export PRIVATE onnx_trt ${OpenCV_LIBS})

#add_executable(infer infer.cpp)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(infer PRIVATE onnx_trt ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})
