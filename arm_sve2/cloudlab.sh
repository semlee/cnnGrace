#!/bin/bash

# Compile individual source files
g++ -g -c main.cpp -o main.o -std=c++11
g++ -g -c ms1/naive_ms1.cpp -o naive_ms1.o -std=c++11
g++ -g -c ms2/regblock_ms2.cpp -o regblock_ms2.o -std=c++11

# Link object files to create the executable
g++ -g main.o naive_ms1.o regblock_ms2.o -o conv_layer -O3 -std=c++11
