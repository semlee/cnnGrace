#!/bin/bash

# Compile individual source files
g++ -c main.cpp -o main.o -fopenmp -std=c++11
g++ -c ms1/naive_ms1.cpp -o naive_ms1.o -fopenmp -std=c++11
g++ -c ms2/regblock_ms2.cpp -o regblock_ms2.o -fopenmp -std=c++11
# g++ -c ms3/regsve_ms3.cpp -o regsve_ms3.o -march=native -std=c++11

# Link object files to create the executable
# g++ main.o naive_ms1.o regblock_ms2.o regsve_ms3.o -o conv_layer -march=native -O3 -std=c++11
g++ main.o naive_ms1.o regblock_ms2.o -o conv_layer -O3 -fopenmp -mavx2 -std=c++11

# Run with perf, focusing on a specific function
# perf record -g -- 

#./conv_layer iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
./conv_layer 1000 224 224 10 3 64 7 7 3 3 2 4 7 7 'F' 'L' 1

rm naive_ms1.o 
rm regblock_ms2.o
rm main.o
rm conv_layer