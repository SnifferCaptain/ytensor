cmake_minimum_required(VERSION 3.10)

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")
set(use_blas 0)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp=libgomp")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma")

project(qwen3_chat)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
endif()

option(YT_USE_LIB "Use prebuilt ytensor shared library backend" OFF)
set(YTENSOR_LIB_ROOT "" CACHE PATH "Path to ytensor/lib directory (contains include/ and bin/)")

set(SRC
	main.cpp
	src/qwen3.cpp
	src/tokenlizer.cpp
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/../../single-header
)

target_link_libraries(${PROJECT_NAME}
	ZLIB::ZLIB
	OpenMP::OpenMP_CXX
	pthread
)

if(YT_USE_LIB)
	if(NOT YTENSOR_LIB_ROOT)
		set(YTENSOR_LIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../lib")
	endif()
	target_compile_definitions(${PROJECT_NAME} PRIVATE YT_USE_LIB)
	target_include_directories(${PROJECT_NAME} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/../../
	)
	target_include_directories(${PROJECT_NAME} PRIVATE
		${YTENSOR_LIB_ROOT}/include
	)
	target_link_directories(${PROJECT_NAME} PRIVATE
		${YTENSOR_LIB_ROOT}/bin
	)
	target_link_libraries(${PROJECT_NAME} ytensor)
endif()

if(use_blas)
	find_package(BLAS REQUIRED)
	target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_USE_BLAS)
	target_include_directories(${PROJECT_NAME} PRIVATE "/usr/include/x86_64-linux-gnu")
	target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
endif()
