#!/bin/bash

# g++ -o naive_layer -mavx2 -O3 run_naive_ms1.cpp -std=c++11
g++ -mavx2 -O3 run_naive_ms1.cpp -o naive_layer_avx2 -std=c++11
g++ -o naive_layer2_native -march=native -mtune=native -O3 run_naive_ms1.cpp -std=c++11
g++ -O3 run_naive_ms1.cpp -o naive_layer_naive -std=c++11

# objdump -D -Mintel naive_layer | grep "vex.v"

echo "-O3"
./naive_layer_naive 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
echo "-mavx2 -O3"
./naive_layer_avx2 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0
echo "-march=native -mtune=native"
./naive_layer2_native 1000 16 16 1 3 64 3 3 1 1 1 'F' 'L' 0

rm naive_layer_naive
rm naive_layer_avx2
rm naive_layer2_native