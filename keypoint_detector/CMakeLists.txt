cmake_minimum_required(VERSION 3.2)

project(keypoint_detector)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.7 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (keypoint_detector src/keypoint_detector.cpp)
target_link_libraries (keypoint_detector ${OpenCV_LIBRARIES})
