#!/bin/bash

set -x

export OMP_NUM_THREADS=10
export KMP_AFFINITY="granularity=fine,proclist=[0-19],explicit"

#./run_resnet50.sh 10 1000 1 f32 F L 1 >> headers.output
./echo_resnet50.sh 10 1000 1 f32 F L 1 | grep PERFDUMP >> resnet3.output
./single_resnet50.sh 10 1000 -1 f32 F L 1 | grep PERFDUMP >> mod_resnet3.output
set +x
