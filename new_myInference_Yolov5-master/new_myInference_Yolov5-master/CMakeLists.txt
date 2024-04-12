cmake_minimum_required(VERSION 3.16)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)
# enable_language(CUDA)
find_package(CUDA REQUIRED cudart cublas curand)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)
#set(CMAKE_LIBRARY_PATH "/usr/local/cuda-11.4/lib64;/home/nvidia/TensorRT-8.2.5.1/lib")

include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

# add_library(myplugins SHARED ${PLUGIN_SRCS})
cuda_add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(yolov5_det ${SRCS})
cuda_add_executable(yolov5_det ${SRCS})
target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
target_link_libraries(yolov5_det myplugins)
target_link_libraries(yolov5_det ${OpenCV_LIBS})


# cmake_minimum_required(VERSION 3.10)

# project(yolov5)

# # Thêm định nghĩa C++11
# set(CMAKE_CXX_STANDARD 11)

# # Thêm định nghĩa cho việc xuất API
# add_definitions(-DAPI_EXPORTS)

# # Tuỳ chọn để sử dụng thư viện CUDA Runtime tĩnh
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# # Xác định trình biên dịch CUDA
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)

# # Bật ngôn ngữ CUDA
# enable_language(CUDA)

# # Thêm thư mục include cho CUDA và OpenCV
# include_directories(
#     /usr/local/cuda-11.4/include
#     ${OpenCV_INCLUDE_DIRS}
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/plugin/
# )

# # Thêm thư viện CUDA và TensorRT
# link_directories(
#     /usr/local/cuda-11.4/lib64
#     /home/nvidia/TensorRT-8.2.5.1/lib/
# )

# # Đặt đường dẫn thư viện
# set(CMAKE_LIBRARY_PATH "/usr/local/cuda-11.4/lib64;/home/nvidia/TensorRT-8.2.5.1/lib")

# # Tìm và đính kèm các thư viện OpenCV
# find_package(OpenCV REQUIRED)

# # Thu thập tất cả các tệp nguồn trong thư mục src và plugin
# file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
# file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

# # Tạo thư viện chia sẻ cho các plugin CUDA
# add_library(myplugins SHARED ${PLUGIN_SRCS})
# target_link_libraries(myplugins nvinfer cudart)

# # Tạo một ứng dụng thực thi cho việc xác định đối tượng YOLOv5
# add_executable(yolov5_det ${SRCS})
# target_link_libraries(yolov5_det nvinfer cudart myplugins ${OpenCV_LIBS})
