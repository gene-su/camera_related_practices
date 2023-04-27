cmake_minimum_required(VERSION 3.2)

project(gradient_magnitude)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (gradient_magnitude src/gradient_magnitude.cpp)
target_link_libraries (gradient_magnitude ${OpenCV_LIBRARIES})