#!/bin/bash

# Compile with debugging information
g++ -g -o regblock_layer -O3 regblock_ms2.cpp -std=c++11

# Run with perf, focusing on a specific function
perf record -g -- ./regblock_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

# Analyze the collected data, focusing on the specific function
perf report --symbol=naive_conv_fp
