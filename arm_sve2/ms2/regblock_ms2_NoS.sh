#!/bin/bash

# # Compile with debugging information
# g++ -g -o regblock_layer -O3 regblock_ms2.cpp -std=c++11

# # Run with perf, focusing on a specific function
# perf record -g -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

# # Analyze the collected data, focusing on the specific function

# perf annotate --symbol=arm_sve_conv_fp

# sudo perf stat -e cache-references,cache-misses,instructions,cycles,branches,branch-misses -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

# perf stat -e instructions,cycles,branches,branch-misses -- ./your_binary 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

# perf record -e branch-misses -- ./your_binary 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
# perf report


# perf record -e cycles -c 1 -ag -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0


g++ -g -O3 run_regblock_ms2.cpp -o reg_layer_naive -std=c++11
# g++ -mavx2 -O2 run_regblock_ms2.cpp -o reg_layer_avx2 -std=c++11
# g++ -o reg_layer_native -march=native -mtune=native -O2 run_regblock_ms2.cpp -std=c++11


# objdump -D -Mintel naive_layer | grep "vex.v"

echo "-O3"
./reg_layer_naive 1000 16 16 5 5 64 3 3 1 1 1 'F' 'L' 0
# echo "-mavx2 -O3"
# ./reg_layer_avx2 10 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
# echo "-march=native -mtune=native"
# ./reg_layer_native 10 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

rm reg_layer_naive
# rm reg_layer_avx2
# rm reg_layer_native