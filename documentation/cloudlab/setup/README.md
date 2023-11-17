## Basic setup using Cloudlab machine

0. Run basic configuration of ubuntu + resize

```
sudo apt update
​
sudo ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime
sudo apt install -y tzdata
sudo dpkg-reconfigure --frontend noninteractive tzdata
​
DEBIAN_FRONTEND=noninteractive sudo -E apt -o DPkg::Options::=--force-confdef upgrade -y
​
sudo apt  install -y htop

sudo sed -e '/swap/ s/^#*/#/' -i /etc/fstab
sudo swapoff /dev/sda3
​
sudo dd if=/dev/zero of=/swapfile bs=1024 count=3145728
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
​
sudo fdisk /dev/sda <<EOF
p
d
2
d
3
d
4
d
n
​
​
​
​
a
p
w
EOF
​
sudo resize2fs /dev/sda1
​
df -H
```
1. pull libxsmm github and restore original version corresponding to paper
```
git clone https://github.com/libxsmm/libxsmm.git. 
git checkout f0af00783745f0382f251bed4ed53d731b775224
```

2. Run basic setup of libraries as below:
(you may create a bash file to run the installation)
```
#!/bin/bash

# Update the package list and upgrade existing packages
sudo apt update
sudo apt upgrade -y

# Install Clang (LLVM)
sudo apt install clang -y

# Install MPI library (OpenMPI)
sudo apt install openmpi-bin -y

# Install OpenCV
sudo apt install libopencv-dev -y

# Install Protobuf
sudo apt install libprotobuf-dev protobuf-compiler -y

# Install Boost
sudo apt install libboost-all-dev -y

# Install LMDB
sudo apt install liblmdb-dev -y

# Install a BLAS library (for example, OpenBLAS)
sudo apt install libopenblas-dev -y
```

3. run folloing code below
```
make realclean && AVX=2 OMP=1 STATIC=1
cd samples/deeplearning/cnnlayer
make realclean && make AVX=2 OMP=1 STATIC=1
```

4. You are good to go.

## Basic experiment on CNN (resnet, layer_example)

1. run_resnet50.sh
You may simply run run_resnet50.sh with correct parameters or use the following code as a bash script:

A. Original Code from Paper
```
export OMP_NUM_THREADS=10
export KMP_AFFINITY=granularity=fine,compact,1,0
./run_resnet50.sh 10 1000 1 f32 F L 1
./run_resnet50.sh 10 1000 1 f32 B L 1
./run_resnet50.sh 10 1000 1 f32 U L 1
./run_googlenetv3.sh 10 1000 1 f32 F L 1
./run_googlenetv3.sh 10 1000 1 f32 B L 1
./run_googlenetv3.sh 10 1000 1 f32 U L 1
```

B. Simplified version
```
#!/bin/bash

set -x

export OMP_NUM_THREADS=10
export KMP_AFFINITY=granularity=fine,compact,1,0

./run_resnet50.sh 10 1000 1 f32 F L 1 | grep PERFDUMP >> resnet.output

set +x
```

2. run single layer
You can also run a single layer 

A. (Additional) running gcc compilation (if makefile does not work)
```
gcc -g -o layer_example_f32 -I/your/path/libxsmm/include layer_example_f32.c \
-L/your/path/libxsmm/lib -lxsmm -lblas -lm -lpthread -fopenmp -ldl
```

B. Run compiled file
```
./layer_example_f32 1000 28 28 40 128 128 3 3 1 1 1 F L 1
```
