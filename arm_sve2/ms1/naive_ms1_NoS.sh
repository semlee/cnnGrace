#!/bin/bash

g++ -o naive_layer -O3 run_naive_ms1.cpp -std=c++11

hexdump -C naive_layer

./naive_layer 1000 16 16 1 3 64 3 3 1 1 1 'A' 'L' 0