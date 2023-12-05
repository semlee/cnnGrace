#!/bin/bash

#SBATCH -t 0-00:30:00
#SBATCH -J conv_layer
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1

iters=1000
ifw=16
ifh=16
nImg=1
nIfm=3
nOfm=64
kw=3
kh=3
padw=1
padh=1
stride=1
type='F'
format='L'
padding_mode=0

# Compile the C++ file
g++ -o conv_layer -O3 main.cpp -std=c++11

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"

    # Run the compiled program with command-line arguments using srun
    #./naive_ms1 iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
    srun -N 1 -p cg1-high --exclusive ./conv_layer $iters $ifw $ifh $nImg $nIfm $nOfm $kw $kh $padw $padh $stride $type $format $padding_mode

    # Optionally, you can pass command-line arguments stored in a file
    # srun -N 1 -p cg1-high --exclusive ./naive_layer $(cat input_args.txt)
else
    echo "Compilation failed"
fi

# Single CG1 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash

# Full CG4 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash