#!/bin/bash

#SBATCH -t 0-00:30:00
#SBATCH -J regblock_run
#SBATCH -o output-%j.out -e output-%j.err
#SBATCH -c 1

iters=1000
ifw=56
ifh=56
nImg=1
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

# Compile the C++ file
g++ -o regblock_layer -O3 run_regblock_ms2.cpp -std=c++11

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"

    # Run the compiled program with command-line arguments using srun
    #./naive_ms1 iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
    srun -N 1 -p cg1-high --exclusive ./regblock_layer $iters $ifw $ifh $nImg $nIfm $nOfm $kw $kh $padw $padh $stride $VLEN $RB_p $RB_q $type $format $padding_mode

    # Optionally, you can pass command-line arguments stored in a file
    # srun -N 1 -p cg1-high --exclusive ./naive_layer $(cat input_args.txt)
else
    echo "Compilation failed"
fi

rm regblock_layer
# Single CG1 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash

# Full CG4 node run
# srun -N 1 -p cg1-high  --exclusive --pty /bin/bash