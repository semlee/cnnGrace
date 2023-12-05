#!/bin/bash

# Compile with debugging information
g++ -g -o conv_layer -O3 main.cpp -std=c++11

# Run with perf, focusing on a specific function
perf record -g -- ./conv_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0