cmake_minimum_required(VERSION 3.2)

project(keypoint_descriptor)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.7 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (keypoint_descriptor src/keypoint_descriptor.cpp)
target_link_libraries (keypoint_descriptor ${OpenCV_LIBRARIES})