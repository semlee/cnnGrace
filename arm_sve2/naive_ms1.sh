#!/bin/bash


#SBATCH -p cg1
#SBATCH -t 0-00:30:00
#SBATCH -J naive_run
#SBATCH -o output-%j.out -e output-%j.err

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
type='A'
format='L'
padding_mode=0

# Compile the C++ file
g++-12.3 -o naive_layer naive_ms1.cpp -std=c++11


# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful"

    # Run the compiled program with command-line arguments
    #./naive_ms1 iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
    ./naive_layer $iters $ifw $ifh $nImg $nIfm $Ofm $kw $kh $padw $padh $stride $type $format $padding_mode

    # Optionally, you can pass command-line arguments stored in a file
    # ./naive_layer $(cat input_args.txt)
else
    echo "Compilation failed"
fi
