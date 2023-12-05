#!/bin/bash

# Compile with debugging information
g++ -g -o regblock_layer -O3 regblock_ms2.cpp -std=c++11

# Run with perf, focusing on a specific function
perf record -g -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

# Analyze the collected data, focusing on the specific function

perf annotate --symbol=arm_sve_conv_fp

sudo perf stat -e cache-references,cache-misses,instructions,cycles,branches,branch-misses -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

perf stat -e instructions,cycles,branches,branch-misses -- ./your_binary 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

perf record -e branch-misses -- ./your_binary 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
perf report


perf record -e cycles -c 1 -ag -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
