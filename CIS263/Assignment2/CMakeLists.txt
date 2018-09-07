cmake_minimum_required(VERSION 3.8)
project(Assignment2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp MyStack.h parr.cpp)
set(CURL_LIBRARY "-lcurl")
find_package(CURL REQUIRED)
add_executable(Assignment2 ${SOURCE_FILES})

include_directories(${CURL_INCLUDE_DIR})

target_link_libraries(Assignment2 ${CURL_LIBRARIES})