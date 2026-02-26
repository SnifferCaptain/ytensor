cmake_minimum_required(VERSION 3.10)

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")
set(use_blas 0)
option(YT_USE_KOMPUTE "Enable Kompute backend" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp=libgomp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")# 使用gomp解决eigen与omp的冲突
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

# 设置编译模式
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma")

project(ymodel2_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

if(YT_USE_KOMPUTE)
    find_path(KOMPUTE_INCLUDE_DIR kompute/Kompute.hpp)
    find_library(KOMPUTE_LIBRARY NAMES kompute Kompute)
    if(NOT KOMPUTE_INCLUDE_DIR OR NOT KOMPUTE_LIBRARY)
        message(FATAL_ERROR "YT_USE_KOMPUTE=ON but Kompute headers/library were not found")
    endif()
endif()

set(SRC
    main.cpp
    ymodel2.cpp
    tokenlizer.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
add_executable(ymodel2_parity_test parity_test.cpp ymodel2.cpp)

# if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     target_compile_options(${PROJECT_NAME} PRIVATE -Wno-pass-failed)
# endif()

target_link_libraries(${PROJECT_NAME}
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
    pthread
)
target_link_libraries(ymodel2_parity_test
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
    pthread
)

if(YT_USE_KOMPUTE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE YT_USE_KOMPUTE=1)
    target_compile_definitions(ymodel2_parity_test PRIVATE YT_USE_KOMPUTE=1)
    target_include_directories(${PROJECT_NAME} PRIVATE ${KOMPUTE_INCLUDE_DIR})
    target_include_directories(ymodel2_parity_test PRIVATE ${KOMPUTE_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${KOMPUTE_LIBRARY})
    target_link_libraries(ymodel2_parity_test ${KOMPUTE_LIBRARY})

    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        set(KOMPUTE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
        set(KOMPUTE_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)
        file(MAKE_DIRECTORY ${KOMPUTE_SPV_DIR})
        add_custom_command(
            OUTPUT ${KOMPUTE_SPV_DIR}/elementwise_add.comp.spv
            COMMAND ${GLSLANG_VALIDATOR} -V -D -S comp -e main -o ${KOMPUTE_SPV_DIR}/elementwise_add.comp.spv ${KOMPUTE_SHADER_DIR}/elementwise_add.comp.hlsl
            DEPENDS ${KOMPUTE_SHADER_DIR}/elementwise_add.comp.hlsl
            COMMENT "Compiling HLSL shader to SPIR-V for Kompute"
            VERBATIM
        )
        add_custom_target(kompute_spirv ALL DEPENDS ${KOMPUTE_SPV_DIR}/elementwise_add.comp.spv)
        add_dependencies(${PROJECT_NAME} kompute_spirv)
    endif()
endif()

if(use_blas)
    # 使用OpenBLAS
    # set(BLA_VENDOR OpenBLAS)

    # 使用MKL
    # set(BLA_VENDOR Intel10_64lp)
    # target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_USE_MKL_ALL)
    # target_include_directories(${PROJECT_NAME} PRIVATE "/usr/include/mkl")

    # 使用系统默认BLAS
    # set(BLA_VENDOR Generic)

    find_package(BLAS REQUIRED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_USE_BLAS)
    target_include_directories(${PROJECT_NAME} PRIVATE "/usr/include/x86_64-linux-gnu")
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
endif()
