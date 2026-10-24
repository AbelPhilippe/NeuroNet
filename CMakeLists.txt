
cmake_minimum_required(VERSION 3.20)

project(NeuroNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(NeuroNet main.cpp)

target_include_directories(NeuroNet PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(NeuroNet PRIVATE
    "${TORCH_LIBRARIES}"
    "${ONNXRUNTIME_LIBRARY}"
)

set_target_properties(NeuroNet PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
