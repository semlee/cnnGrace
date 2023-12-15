#!/bin/bash

#SBATCH -t 0-00:30:00
#SBATCH -J conv_layer
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 72

iters=10
ifw=56
ifh=56
nImg=72
nIfm=64
nOfm=256
kw=1
kh=1
padw=0
padh=0
stride=1
VLEN=4 #128bit = 32bit * 4
RB_p=7
RB_q=7
type='F'
format='L'
padding_mode=1

export OMP_NUM_THREADS=72
# Compile individual source files
g++ -c main.cpp -o main.o -fopenmp -std=c++11
g++ -c ms1/naive_ms1.cpp -o naive_ms1.o -fopenmp -std=c++11
g++ -c ms2/regblock_ms2.cpp -o regblock_ms2.o -fopenmp -std=c++11
# g++ -c ms3/regsve_ms3.cpp -o regsve_ms3.o -march=native -std=c++11

# Link object files to create the executable
# g++ main.o naive_ms1.o regblock_ms2.o regsve_ms3.o -o conv_layer -march=native -O3 -std=c++11
g++ main.o naive_ms1.o regblock_ms2.o -o conv_layer -O3 -fopenmp -std=c++11

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"

    # Run the compiled program with command-line arguments using srun
    #./naive_ms1 iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
    srun -N 1 -p cg1-cpu480gb-gpu96gb --exclusive ./conv_layer $iters $ifw $ifh $nImg $nIfm $nOfm $kw $kh $padw $padh $VLEN $RB_p $RB_q $stride $type $format $padding_mode

    # Optionally, you can pass command-line arguments stored in a file
    # srun -N 1 -p cg1-high --exclusive ./naive_layer $(cat input_args.txt)
else
    echo "Compilation failed"
fi

# Single CG1 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash

# Full CG4 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash

rm main.o
rm naive_ms1.o
rm regblock_ms2.o
# rm regsve_ms3.o
rm conv_layer