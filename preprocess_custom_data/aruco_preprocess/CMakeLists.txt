project(PieceSeg)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibration calibration.cpp)
target_link_libraries(calibration ${OpenCV_LIBS} cnpy)
