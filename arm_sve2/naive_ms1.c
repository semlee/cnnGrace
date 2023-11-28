#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Data fetched from original layer_example_f32
/* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 0;           /* padding in input, height */
  int padw = 0;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  int padding_mode = 0;   /* padding mode */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'A';      /* 'A': ALL, 'L': LIBXSMM, 'T': Tensorflow, 'M', Mixed */


// Algorithm 1: Naive Algorithm using OpenMP

/*
    N = # of input image (NImg)
    K = # of output feature map (Nof)
    C = # of input feature map (Nif)
    H = size of input/output feature map (Nix/Nox)
    W = size of input/output feature map (Niy/Noy)
    P = (H + 2 * padding - R) / (stride + 1)
    Q = (Q + 2 * padding - S) / (stride + 1)
    R = Kernel height (Kx)
    S = kernel width (Ky)
    stride
    padding

    4 Dimensional Arrays :
    O = output image
    I = input image
    W = weight
*/

int P = (H + 2 * padding - R) / (stride + 1)
int Q = (Q + 2 * padding - S) / (stride + 1)

//Naive Forward Propagation Loops
/*
for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int oj = 0; oj < P; oj++) {
                for (int oi = 0; oi < Q; oi++) {
                    int ij = stride * oj - padding;
                    int ii = strid * oi - padding;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            O[n][k][oj][oi] += I[n][c][ij + r][ii + s] ∗ W[k][c][r][s]
                        }
                    }

                }
            }
        }
    }
}
*/

void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias) {
#if defined (OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    for (img = 0; img < nImg; ++img) {
        for (ofm = 0; ofm < nOfm; ++ofm) {
            for (ifm = 0; ifm < nIfm; ++ifm) {
                for (oj = 0; oj < ofh; ++oj) {
                    ij = oj * stride_h - pad_h;
                    for (oi = 0; oi < ofw; ++oi) {
                        ii = oi * stride_w - pad_w;
                        for (kj = 0; kj < kh; ++kj) {
                            if (ij+kj < 0 || ij+kj >= ifh) continue;
                            for (ki = 0; ki < kw; ++ki) {
                                if (ii+ki < 0 || ii+ki >= ifw) continue;
                                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
                                output[n][k][oj][oi] += input[n][c][ij + r][ii + s] ∗ filter[k][c][r][s];
                            }
                        }
                    }
                }
            }
        }
    }

    
}


//Naive Backward Propagation Loops
#pragma omp parallel for
for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int oj = 0; oj < P; oj ++) {
                for (int oi = 0; oi < Q; oi++) {
                    int ij = stride * oj;
                    int ii = stride * oi;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            dI[n][c][ij+r][ii+s] += dO[n][k][oj][oi] * W[k][c][r][s];
                        }
                    }
                }
            }
        }
    }
}

//Naive Weight Update Gradient Loops
#pragma omp parallel for
for (int i = 0; i < N; n++) {
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int oj = 0; oj < P; oj ++) {
                for (int oi = 0; oi < Q; oi++) {
                    int ij = stride * oj;
                    int ii = stride * oi;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            dW[k][c][r][s] += I[n][c][ij+r][ii+s] * dO[n][k][oj][oi];
                        }
                    }
                }
            }
        }
    }
}
