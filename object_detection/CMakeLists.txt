cmake_minimum_required(VERSION 3.2)

project(object_detection)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.7 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (object_detection src/object_detection.cpp)
target_link_libraries (object_detection ${OpenCV_LIBRARIES})