3

# Run with perf, focusing on a specific function
# perf record -g -- 

#./conv_layer iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padding_mode
./conv_layer 1000 16 16 5 32 64 3 3 1 1 1 'F' 'L' 0

rm naive_ms1.o 
rm regblock_ms2.o
rm main.o
rm conv_layer