cmake_minimum_required(VERSION 3.16)

project(ytensor_lib_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

set(YT_LIB_ROOT ${CMAKE_CURRENT_LIST_DIR})

add_library(ytensor SHARED
    ${YT_LIB_ROOT}/src/ytensor_library.cpp
)

target_include_directories(ytensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ytensor PUBLIC
    OpenMP::OpenMP_CXX
    ZLIB::ZLIB
)

if(WIN32)
    set_target_properties(ytensor PROPERTIES
        OUTPUT_NAME ytensor
        RUNTIME_OUTPUT_DIRECTORY ${YT_LIB_ROOT}/bin
        LIBRARY_OUTPUT_DIRECTORY ${YT_LIB_ROOT}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${YT_LIB_ROOT}/bin
    )
else()
    set_target_properties(ytensor PROPERTIES
        OUTPUT_NAME ytensor
        LIBRARY_OUTPUT_DIRECTORY ${YT_LIB_ROOT}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${YT_LIB_ROOT}/bin
    )
endif()
