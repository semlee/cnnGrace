#!/bin/bash

set -x

export OMP_NUM_THREADS=10
export KMP_AFFINITY="granularity=fine,compact,1,2"

./echo_resnet50.sh 10 1000 1 f32 F L 1 | grep PERFDUMP >> resnet.output
./single_resnet50.sh 10 1000 -1 f32 F L 1 | grep PERFDUMP >> single.output

set +x
