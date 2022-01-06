find_package(PythonInterp REQUIRED)
cmake_minimum_required(VERSION 3.5)
project(libtorch_yolox)

set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED PATHS /home/song/deploy/torchscript-mnist/libtorch)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch_yolox libtorch_yolox.cpp)
target_link_libraries(libtorch_yolox ${TORCH_LIBRARIES} ${OpenCV_LIBS})
