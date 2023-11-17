#### This is a documentation page for informations regarding the tests run by cloudlab, and Grace

## Cloudlab Cluster Specification

|     | feature |
| --- | --- |
| CPU name | four Intel(R) Xeon(R) CPU E5-2660 v3 |
| Architecture | x86-64 |
| Core count | 40 cores (10 each) |
| Threads per core | 2 threads |
| SIMD support | AVX2 |
| L1 cache | 640KiB i-cache + 640KiB d-cache| 
| L2 cache | 5MiB per core | 
| L3 cache | 50 MiB (25MB shared memory) | 
| DDR4 size | 768 GB |
| Memory bandwidth | upto 64GB/s |

## NVIDIA Grace CPU Specification
Whitepaper:
https://resources.nvidia.com/en-us-grace-cpu/nvidia-grace-cpu-superchip#page=1

|     | feature |
| --- | --- |
| CPU name | Arm Neoverse V2 |
| Architecture | Arm64 |
| Core count | 144 cores |
| SIMD support | 4x128b SVE2 |
| L1 cache | 64KB i-cache + 64KB d-cache| 
| L2 cache | 1MB per core| 
| L3 cache | 234MB| 
| LPDDR5X size | 240GB, 480GB and 960GB on-module memory options | 
| Memory bandwidth | Up to 1TB/s |
| NVIDIA NVLink-C2C bandwidth | 900GB/s |
| PCIe | links Up to 8x PCIe Gen5 x16 option to bifurcate |
| Module thermal design power (TDP) | 500W TDP with memory |
| Form factor | Superchip module |
| Thermal solution | Air cooled or liquid cooled |

## Several Items to fulfill the project

### Grace requirement to run libxsmm conv_layer
1. GCC / Clang Compiler 
2. OpenMPI
3. OpenCV
4. Protobuf
5. Boost
6. LMDB
7. OpenBlas

### Libraries and / or methodologies to run libxsmm or small matmul using ARM
1. Arm Compute Library
2. SVE-optimized BLAS libraries
3. Compiler auto-vectorization
4. use SVE intrinsics to code libxsmm

### Intrinsic functions utilized on AVX2 microkernels
```
libxsmm_x86_instruction_vec_move
libxsmm_x86_instruction_alu_imm
libxsmm_x86_instruction_vec_compute_reg
```

### Intrinsic functions utilized on AVX512 microkernels 
* based on generator_gemm_avx512_microkernel.h or .c file
* detail can be found on generator_x86_instruction.h or .c file
```
libxsmm_x86_instruction_alu_imm
libxsmm_x86_instruction_alu_reg
libxsmm_x86_instruction_vec_compute_reg
libxsmm_x86_instruction_vec_move
libxsmm_x86_instruction_prefetch
libxsmm_x86_instruction_vec_compute_mem
libxsmm_x86_instruction_vec_compute_qfma
```
defined integer as below can be found on generator_common.h file

## General Milestone for converting AVX SIMD CNN to SVE2 SIMD CNN
#### 1. Download all of the components required for running libxsmm on Grace
#### 2. Try out using normal matrix multiplication library, instead of libxsmm on ARM
#### 3. Test out libxsmm-like function using provided library on ARM (Target Goal)
#### 4. convert libxsmm_x86_instructions using SVE2 intrinsics (described above)
* libxsmm/src/generator_x86_instructions.h
* libxsmm/src/generator_x86_instructions.c
* libxsmm/src/generator_common.c
* libxsmm/src/generator_common.h
* libxsmm/include/libxsmm_x86_intrinsics.h
#### 5. generate separate Makefile using ARM suitable code

## Detailed Milestone
* MS1. Naive Direct Conv. onto Grace  
Use OpenMP, simple arithmetic (+=, *)  
* MS2. Run Register Blocking (Algorithm2) + using SVE2 Intrinsics  
    a. SVE2 (No FMA)  
https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[sve2]&f:@navigationhierarchiesinstructiongroup=[Vector%20arithmetic,Multiply-accumulate,Saturating%20multiply-accumulate]&f:@navigationhierarchieselementbitsize=[32]  
    b. SVE (FMA)  
https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiesinstructiongroup=[Vector%20arithmetic,Multiply-accumulate,Fused%20multiply-accumulate]&f:@navigationhierarchiessimdisa=[sve]&f:@navigationhierarchieselementbitsize=[32]  
* MS3. Microkernel implementation using JIT-ed Comilation  
* MS4. Fused Layer implementation  
* MS5. Add Dryrun and Replay phase  

