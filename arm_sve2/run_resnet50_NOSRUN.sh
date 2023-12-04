#!/bin/bash

# Compile the C++ file
g++ -o naive_layer naive_ms1.cpp -std=c++11

ITERS=1000
MB=1
TYPE='A'
FORMAT='L'
PAD=1

# Define arrays for each parameter
ifw_values=(224 56 56 56 56 56 28 28 28 28 28 14 14 14 14 14 7)
ifh_values=(224 56 56 56 56 56 28 28 28 28 28 14 14 14 14 14 7)
nImg_values=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
nIfm_values=(3 64 64 256 256 128 128 512 128 512 512 256 256 1024 256 512 512)
nOfm_values=(64 256 64 64 256 128 128 512 512 128 512 256 256 512 256 1024 512)
kw_values=(7 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3 3)
kh_values=(7 1 1 3 1 1 3 1 1 3 1 1 3 1 1 3 3)
padw_values=(3 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1)
padh_values=(3 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1)
stride_values=(2 1 1 1 2 2 1 1 1 2 2 1 2 1 1 1 1)

# Specify the number of cores
NUM_CORES=1

# Iterate over the indices of the arrays
for i in "${!ifw_values[@]}"; do
    ./naive_layer \
        -c $NUM_CORES \
        $ITERS ${ifw_values[$i]} ${ifh_values[$i]} ${nImg_values[$i]} ${nIfm_values[$i]} ${nOfm_values[$i]} \
        ${kw_values[$i]} ${kh_values[$i]} ${padw_values[$i]} ${padh_values[$i]} ${stride_values[$i]} \
        $TYPE $FORMAT $PAD
done