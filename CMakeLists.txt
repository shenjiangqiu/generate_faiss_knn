cmake_minimum_required(VERSION 3.10)
project(MyDemo)

# Set the path to your include directory
include_directories(/home/sjq/miniconda3/envs/faiss_env/include/)

# Set the path to your library directory
link_directories(/home/sjq/miniconda3/envs/faiss_env/lib)

# Add all files in demo/ directory

# Add executable for each source file
    # Extract the filename without extension
add_executable(main main.cc)
    # Link against faiss library
target_link_libraries(main faiss_avx2)
