cmake_minimum_required(VERSION 3.2)

project(harris_corner)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (harris_corner src/harris_corner.cpp)
target_link_libraries (harris_corner ${OpenCV_LIBRARIES})
