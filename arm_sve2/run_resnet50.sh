#!/bin/bash

#SBATCH -t 0-00:30:00
#SBATCH -J run_resnet50
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1

# Compile the C++ file
# Compile individual source files
g++ -c main.cpp -o main.o -std=c++11
g++ -c ms1/naive_ms1.cpp -o naive_ms1.o -std=c++11
g++ -c ms2/regblock_ms2.cpp -o regblock_ms2.o -std=c++11
g++ -c ms3/regsve_ms3.cpp -o regsve_ms3.o -march=native -std=c++11
# Link object files to create the executable
g++ main.o naive_ms1.o regblock_ms2.o regsve_ms3.o -o conv_layer -march=native -O3 -std=c++11

ITERS=1000
MB=1
TYPE='F'
FORMAT='L'
PAD=1
VLEN=4
RB_p=1
RB_q=1

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

# Iterate over the indices of the arrays
for i in "${!ifw_values[@]}"; do
    srun -N 1 -p cg1-high --exclusive ./conv_layer \
        $ITERS ${ifw_values[$i]} ${ifh_values[$i]} ${nImg_values[$i]} ${nIfm_values[$i]} ${nOfm_values[$i]} \
        ${kw_values[$i]} ${kh_values[$i]} ${padw_values[$i]} ${padh_values[$i]} ${stride_values[$i]} \
        $VLEN $RB_p $RB_q $TYPE $FORMAT $PAD
done