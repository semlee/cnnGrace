#!/bin/bash

# Compile individual source files
g++ -c main.cpp -o main.o -std=c++11
g++ -c ms1/naive_ms1.cpp -o naive_ms1.o -std=c++11
g++ -c ms2/regblock_ms2.cpp -o regblock_ms2.o -std=c++11

# Link object files to create the executable
g++ main.o naive_ms1.o regblock_ms2.o -o conv_layer -O3 -std=c++11

# Run with perf, focusing on a specific function
perf record -g -- ./conv_layer 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0