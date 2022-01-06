cmake_minimum_required(VERSION 3.5)
project(yolov3)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(yolov3 demo_video.cpp)
target_link_libraries(yolov3 ${OpenCV_LIBS})
