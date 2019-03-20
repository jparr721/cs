cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(slam LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
find_package(OpenCV REQUIRED)

add_executable(slam slam.cc view.cc)
target_link_libraries(slam ${OpenCV_LIBS})
