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
# g++ -c ms3/regsve_ms3.cpp -o regsve_ms3.o -march=native -std=c++11
# Link object files to create the executable
g++ main.o naive_ms1.o regblock_ms2.o -o conv_layer -O3 -std=c++11

ITERS=10
MB=1 #OMP_NUM_THREADS
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

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type

# Iterate over the indices of the arrays
for i in "${!ifw_values[@]}"; do
    srun -N 1 -p cg1-high --exclusive ./conv_layer \
        $ITERS ${ifw_values[$i]} ${ifh_values[$i]} $nImg_values ${nIfm_values[$i]} ${nOfm_values[$i]} \
        ${kw_values[$i]} ${kh_values[$i]} ${padw_values[$i]} ${padh_values[$i]} ${stride_values[$i]} \
        $VLEN $RB_p $RB_q $TYPE $FORMAT $PAD
done

rm main.o
rm naive_ms1.o
rm regblock_ms2.o
# rm regsve_ms3.o
rm conv_layer