cmake_minimum_required(VERSION 3.10)
project(ytensor_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp=libgomp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma")

option(YT_USE_KOMPUTE "Enable Kompute backend" OFF)

find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

# --- shader manifest test (no Kompute dependency) ---
add_executable(test_shader_manifest test_shader_manifest.cpp)

# --- GPU dispatch test ---
add_executable(test_gpu_dispatch test_gpu_dispatch.cpp)
target_link_libraries(test_gpu_dispatch ZLIB::ZLIB OpenMP::OpenMP_CXX pthread)

if(YT_USE_KOMPUTE)
    find_path(KOMPUTE_INCLUDE_DIR kompute/Kompute.hpp)
    find_library(KOMPUTE_LIBRARY NAMES kompute Kompute)
    if(NOT KOMPUTE_INCLUDE_DIR OR NOT KOMPUTE_LIBRARY)
        message(FATAL_ERROR "YT_USE_KOMPUTE=ON but Kompute headers/library were not found")
    endif()

    target_compile_definitions(test_gpu_dispatch PRIVATE YT_USE_KOMPUTE=1)
    target_include_directories(test_gpu_dispatch PRIVATE ${KOMPUTE_INCLUDE_DIR})
    target_link_libraries(test_gpu_dispatch ${KOMPUTE_LIBRARY})

    # Compile core HLSL shaders to SPIR-V
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        set(KOMPUTE_CORE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/kompute)
        set(KOMPUTE_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)
        file(MAKE_DIRECTORY ${KOMPUTE_SPV_DIR})

        # Each shader entry point needs its own SPIR-V compilation
        set(SHADER_ENTRIES
            # file                              entry        output name
            "binary_arith_float.comp.hlsl;fadd_main;binary_arith_float_add"
            "binary_arith_float.comp.hlsl;fsub_main;binary_arith_float_sub"
            "binary_arith_float.comp.hlsl;fmul_main;binary_arith_float_mul"
            "binary_arith_float.comp.hlsl;fdiv_main;binary_arith_float_div"
            "binary_arith.comp.hlsl;main;binary_arith"
            "binary_cmp.comp.hlsl;main;binary_cmp"
            "matmul.comp.hlsl;main;matmul"
            "matview.comp.hlsl;main;matview"
            "reduction_sum.comp.hlsl;main;reduction_sum"
            "reduction_max.comp.hlsl;main;reduction_max"
        )

        set(KOMPUTE_SPV_OUTPUTS)
        foreach(ENTRY ${SHADER_ENTRIES})
            list(GET ENTRY 0 HLSL_NAME)
            list(GET ENTRY 1 ENTRY_POINT)
            list(GET ENTRY 2 SPV_NAME)
            set(HLSL_FILE ${KOMPUTE_CORE_SHADER_DIR}/${HLSL_NAME})
            set(SPV_FILE ${KOMPUTE_SPV_DIR}/${SPV_NAME}.spv)
            add_custom_command(
                OUTPUT ${SPV_FILE}
                COMMAND ${GLSLANG_VALIDATOR} -V -D -S comp -e ${ENTRY_POINT} -o ${SPV_FILE} ${HLSL_FILE}
                DEPENDS ${HLSL_FILE}
                COMMENT "Compiling ${HLSL_NAME}:${ENTRY_POINT} -> ${SPV_NAME}.spv"
                VERBATIM
            )
            list(APPEND KOMPUTE_SPV_OUTPUTS ${SPV_FILE})
        endforeach()

        add_custom_target(kompute_spirv ALL DEPENDS ${KOMPUTE_SPV_OUTPUTS})
        add_dependencies(test_gpu_dispatch kompute_spirv)
    endif()
endif()
