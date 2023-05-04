cmake_minimum_required(VERSION 3.2)

project(lidar_points_to_image_projection)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (lidar_points_to_image_projection src/lidar_points_to_image_projection.cpp)
target_link_libraries (lidar_points_to_image_projection ${OpenCV_LIBRARIES})