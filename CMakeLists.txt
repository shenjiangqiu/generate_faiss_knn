cmake_minimum_required(VERSION 3.24)
project(MyDemo)
add_subdirectory("external/faiss")
# Set the path to your include directory
# include_directories(/home/sjq/miniconda3/envs/faiss_env/include/)

# Set the path to your library directory
# link_directories(/home/sjq/miniconda3/envs/faiss_env/lib)

# log the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# test if avx512 is supported
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    message(STATUS "AVX512 support enabled")
else()
    message(STATUS "AVX512 support not enabled")
endif()
# Add all files in demo/ directory

# Add executable for each source file
    # Extract the filename without extension
add_executable(main main.cc)
    # Link against faiss library
if (COMPILER_SUPPORTS_AVX512)
    target_link_libraries(main faiss_avx512)
else()
    target_link_libraries(main faiss_avx2)
endif()
target_include_directories(main PUBLIC .)
