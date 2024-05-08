build:
    cmake -B ./build_release
    cmake --build ./build_release
run base train query gt output:build
    ./build_release/main -b {{base}} -t {{train}} -q {{query}} -g {{gt}} -o {{output}}