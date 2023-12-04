#!/bin/bash

set -x

export OMP_NUM_THREADS=10
export KMP_AFFINITY=granularity=fine,compact,1,0

#./run_resnet50.sh 10 1000 1 f32 F L 1 >> headers.output
./echo_resnet50.sh 10 1000 1 f32 F L 1 | grep PERFDUMP >> resnet2.output
./single_resnet50.sh 10 1000 1 f32 F L 1 | grep PERFDUMP >> mod_resnet.output
./single_resnet50.sh 10 1000 -1 f32 F L 1 | grep PERFDUMP >> mod_resnet2.output
set +x
