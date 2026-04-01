cmake_minimum_required(VERSION 3.16)

project(ytensor_lib_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

set(YT_LIB_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(YT_PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

add_library(ytensor SHARED
    ${YT_LIB_ROOT}/src/ytensor_library.cpp
)

target_include_directories(ytensor
    PUBLIC
        ${YT_LIB_ROOT}/include
)

target_link_libraries(ytensor
    PUBLIC
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

set(YT_SINGLE_HEADER_SRC ${YT_PROJECT_ROOT}/single-header/ytensor_single.hpp)
set(YT_SINGLE_HEADER_DST ${YT_LIB_ROOT}/include/ytensor_single.hpp)

add_custom_command(
    OUTPUT ${YT_SINGLE_HEADER_SRC}
    COMMAND ${CMAKE_COMMAND} -E env python3 ${YT_PROJECT_ROOT}/single-header/packer.py
    WORKING_DIRECTORY ${YT_PROJECT_ROOT}
    DEPENDS
        ${YT_PROJECT_ROOT}/ytensor.hpp
        ${YT_PROJECT_ROOT}/include/ytensor_base.hpp
        ${YT_PROJECT_ROOT}/include/ytensor_core.hpp
        ${YT_PROJECT_ROOT}/include/ytensor_io.hpp
        ${YT_PROJECT_ROOT}/include/ytensor_function.hpp
        ${YT_PROJECT_ROOT}/src/ytensor_base.inl
        ${YT_PROJECT_ROOT}/src/ytensor_base_templates.inl
        ${YT_PROJECT_ROOT}/src/ytensor_io.inl
        ${YT_PROJECT_ROOT}/src/ytensor_io_templates.inl
        ${YT_PROJECT_ROOT}/single-header/packer.py
    COMMENT "Generating single-header/ytensor_single.hpp"
    VERBATIM
)

add_custom_command(
    OUTPUT ${YT_SINGLE_HEADER_DST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${YT_LIB_ROOT}/include
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${YT_SINGLE_HEADER_SRC} ${YT_SINGLE_HEADER_DST}
    DEPENDS ${YT_SINGLE_HEADER_SRC}
    COMMENT "Syncing lib/include/ytensor_single.hpp"
    VERBATIM
)

add_custom_target(ytensor_single_header ALL DEPENDS ${YT_SINGLE_HEADER_DST})
add_dependencies(ytensor ytensor_single_header)

