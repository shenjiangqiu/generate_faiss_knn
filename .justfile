build_debug:
    cmake -B ./build_debug -DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DFAISS_OPT_LEVEL=avx512 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build ./build_debug -j -- main
build_release:
    cmake -B ./build_release -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DFAISS_OPT_LEVEL=avx512 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build ./build_release -j -- main

run base train query gt output:build_release
    ./build_release/main -b {{base}} -t {{train}} -q {{query}} -g {{gt}} -o {{output}}