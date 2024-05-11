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

add_library(common common.cc common.h )
target_include_directories(common PUBLIC . target/cxxbridge)
target_link_directories(common PUBLIC target/release)
target_link_libraries(common faiss_avx512 generate_faiss_knn)

# Define a list of source files
set(executable_sources main_autotune.cc main_selected.cc gt.cc)

# Iterate over each source file and create an executable
foreach(source_file ${executable_sources})
    # Extract the executable name from the source file
    get_filename_component(executable_name ${source_file} NAME_WE)

    # Add the executable target
    add_executable(${executable_name} ${source_file})

    # Link against faiss library
    target_precompile_headers(${executable_name} PRIVATE CLI11.hpp)
    target_link_libraries(${executable_name} common)
    target_include_directories(${executable_name} PUBLIC .)
endforeach()

