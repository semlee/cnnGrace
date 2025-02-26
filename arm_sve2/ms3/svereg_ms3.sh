#!/bin/bash

#SBATCH -t 0-01:00:00
#SBATCH -J regblock_run
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p c2-2x240gb 

iters=1000
ifw=16
ifh=16
nImg=5
nIfm=5
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
g++ -o svereg_layer -O3 -march=native run_svereg_ms3.cpp -std=c++11
# -march=armv9.3-a+sve
# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"

    # Run the compiled program with command-line arguments using srun
    #./naive_ms1 iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
    srun -N 1--exclusive ./svereg_layer $iters $ifw $ifh $nImg $nIfm $nOfm $kw $kh $padw $padh $stride $type $format $padding_mode

    # Optionally, you can pass command-line arguments stored in a file
    # srun -N 1 -p cg1-high --exclusive ./naive_layer $(cat input_args.txt)
else
    echo "Compilation failed"
fi


rm svereg_layer

# Single CG1 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash

# Full CG4 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash