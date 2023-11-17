# Results

## Headers for CSV files
 <!-- ./layer_example_${BIN}             iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type
 ${NUMACTL} ./layer_example_${BIN} ${ITERS}   7        7     ${MB} 2048  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} -->

PERFDUMP,FP,1.9-1623,10,10,2048,512,7,7,1,1,1,0,0,0.0023663,434.26,7520.864536,7520.864536,0.000000,0.000000,0.000000,0.000000,0.000000
|--|--|--| N | N | nIfm | nOfm | inpWidth | inpHeight | kw | kh | stride | kw | kh | fp time | GFLOPs | L1 reference | L1 test |--|--|
```
##########################################
#          Setting Up (Common)           #
##########################################
PARAMS: W:7  H:7  N:10  C:2048  K:512  R:1  S:1  P:7  Q:7  STRIDE:1
PARAMS: ITERS:1000
 InImg 7x7 Padded (7x7)
OutImg 7x7 Padded (7x7)
SIZE Input  (MB):       3.83 MiB
SIZE Output (MB):       0.96 MiB
SIZE Input   (1):       0.38 MiB
SIZE Output  (1):       0.10 MiB
SIZE Weight     :       4.00 MiB
Using Overwrite Option
##########################################
#         Computing Reference ...        #
##########################################
##########################################
#      Computing Reference ... done      #
##########################################

##########################################
#      Setting Up  (custom-Storage)      #
##########################################
##########################################
#   Correctness - FWD (custom-Storage)   #
##########################################
L1 reference  : 7520.864535501550562912598
L1 test       : 7520.864535501550562912598
L2 abs.error  : 0.000000000000000000000000
L2 rel.error  : 0.000000000000000000000000
Linf abs.error: 0.000000000000000000000000
Linf rel.error: 0.000000000000000000000000
Check-norm    : 0.000000000000000000000000
##########################################
#   Performance - FWD (custom-Storage)   #
##########################################
GFLOP  = 1.0276
fp time = 0.0023663
GFLOPS  = 434.26
```

## Basic characteristics of Results
### Naive Run using code from Anatomy paper

MB: 10  
ITERS: 1000  
NUMA: 1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 2  
NC: 20  
NT: 40  
HT: 2  
NN: 2  
OMP_NUM_THREADS: 10  
KMP_AFFINITY: granularity=fine,compact,1,0

### Modified Ver 2. Hardcode size of socket, core and thread with NUMA=-1
MB: 10  
ITERS: 1000  
NUMA: -1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 1  
NC: 10  
NT: 20  
HT: 2  
NN: 1  
OMP_NUM_THREADS: 10  
KMP_AFFINITY: granularity=fine,compact,1,0  

### Mod3 : Naive + OMP_NUM_THREADS  = 20
MB: 20  
ITERS: 1000  
NUMA: 1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 2  
NC: 20  
NT: 40  
HT: 2  
NN: 2  
OMP_NUM_THREADS: 20  
KMP_AFFINITY: granularity=fine,compact,1,0  

### Mod4 : Hardcode + OMP_NUM_THREADS = 20
MB: 20  
ITERS: 1000  
NUMA: -1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 1  
NC: 10  
NT: 20  
HT: 2  
NN: 1  
OMP_NUM_THREADS: 20  
KMP_AFFINITY: granularity=fine,compact,1,0  

### Mod5 : Naive + OMP_NUM_THREAD = 20 + compact 1,2

MB: 20  
ITERS: 1000  
NUMA: 1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 2  
NC: 20  
NT: 40  
HT: 2  
NN: 2  
OMP_NUM_THREADS: 20  
KMP_AFFINITY: granularity=fine,compact,1,2

### Mod6 : Hardcode + OMP_NUM_THREAD = 20 + compact, 1,2

MB: 20  
ITERS: 1000  
NUMA: -1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 1  
NC: 10  
NT: 20  
HT: 2  
NN: 1  
NUMACTL:   
OMP_NUM_THREADS: 20  
KMP_AFFINITY: granularity=fine,compact,1,2


### Mod7 : Naive + OMP_NUM_THREAD = 40 + compact, 1,2

MB: 40  
ITERS: 1000  
NUMA: 1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 2  
NC: 20  
NT: 40  
HT: 2  
NN: 2  
OMP_NUM_THREADS: 40  
KMP_AFFINITY: granularity=fine,compact,1,2  

### Mod8 : Hardcode + OMP_NUM_THREAD = 40 + compact, 1,2
MB: 40  
ITERS: 1000  
NUMA: -1  
BIN: f32  
TYPE: F  
FORMAT: L  
PAD: 1  
NS: 1  
NC: 10  
NT: 20  
HT: 2  
NN: 1  
OMP_NUM_THREADS: 40  
KMP_AFFINITY: granularity=fine,compact,1,2 

