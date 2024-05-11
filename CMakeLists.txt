cmake_minimum_required(VERSION 3.24)
project(MyDemo)
add_subdirectory("external/faiss")
# Set the path to your include directory
# include_directories(/home/sjq/miniconda3/envs/faiss_env/include/)

# Set the path to your library directory
# link_directories(/home/sjq/miniconda3/envs/faiss_env/lib)

# log the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add all files in demo/ directory
# enable wall
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

add_library(common common.cc common.h)
target_include_directories(common PUBLIC .)
target_link_libraries(common faiss)

# Add executable for each source file
    # Extract the filename without extension
add_executable(main main.cc)
    # Link against faiss library
target_precompile_headers(main PRIVATE CLI11.hpp)
target_link_libraries(main faiss common)
target_include_directories(main PUBLIC .)


add_executable(gt gt.cc)
target_link_libraries(gt faiss common)
target_include_directories(gt PUBLIC .)