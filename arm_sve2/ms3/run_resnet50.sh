#!/bin/bash

#SBATCH -t 0-02:00:00
#SBATCH -J run_resnet50
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 72
#SBATCH -N 1
#SBATCH -p cg1-cpu480gb-gpu96gb

# Compile the C++ file
# Compile individual source files
export OMP_NUM_THREADS=72
g++ -o conv_layer -march=native -O3 -fopnemp run_svereg_ms3.cpp -std=c++11

ITERS=100
MB=72
TYPE='F'
FORMAT='L'
PAD=1
VLEN=4 #128bit = 32bit * 4
RB_p=7
RB_q=7

# Define arrays for each parameter
ifw_values=(224 56 56 56 56 56 56 28 28 28 28 28 14 14 14 14 14 7 7 7)
ifh_values=(224 56 56 56 56 56 56 28 28 28 28 28 14 14 14 14 14 7 7 7)
nImg_values=$MB
nIfm_values=(3 64 64 64 256 256 256 128 128 512 512 512 256 256 1024 1024 1024 512 512 2048)
nOfm_values=(64 256 64 64 64 512 128 128 512 128 1024 256 256 1024 256 2048 512 512 2048 512)
kw_values=(7 1 1 3 1 1 1 3 1 1 1 1 3 1 1 1 1 3 1 1)
kh_values=(7 1 1 3 1 1 1 3 1 1 1 1 3 1 1 1 1 3 1 1)
padw_values=(3 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0)
padh_values=(3 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0)
stride_values=(2 1 1 1 1 2 2 1 1 1 2 2 1 1 1 2 2 1 1 1)

# Iterate over the indices of the arrays
for i in "${!ifw_values[@]}"; do
    srun -N 1 --exclusive ./conv_layer \
        $ITERS ${ifw_values[$i]} ${ifh_values[$i]} $nImg_values ${nIfm_values[$i]} ${nOfm_values[$i]} \
        ${kw_values[$i]} ${kh_values[$i]} ${padw_values[$i]} ${padh_values[$i]} ${stride_values[$i]} \
        $VLEN $RB_p $RB_q $TYPE $FORMAT $PAD 
done

#taskset 0x00000001
rm conv_layer