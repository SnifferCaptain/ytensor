cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

project(ad_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)

# Test 1: Basic graph test
add_executable(test_ad_graph test_ad_graph.cpp)
target_link_libraries(test_ad_graph ZLIB::ZLIB pthread)

# Test 2: FFN comparison test
add_executable(test_ad_ffn test_ad_ffn.cpp)
target_link_libraries(test_ad_ffn ZLIB::ZLIB pthread)

