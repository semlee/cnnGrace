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


// Algorithm 2: Optimized Algorithm using OpenMP + Register Blocking

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


int C_b = C/VLEN;
int K_b = K/VLEN;
int P_b = P/RB_p;
int Q_b = Q/RB_q;

//Optimized Forward Propagation Loops : Register Blocking
#pragma omp parallel for
for (int n = 0; n < N; n++) {
    for (int k_b = 0; k_b < K_b; k_b++) {
        for (int c_b = 0; c_b < C_b; c_b++) {
            for (int oj_b = 0; oj_b < P_b; oj_b++) {
                for (int oi_b = 0; oi_b < Q_b; oi_b++) {
                    int ij = stride * oj * RB_p;
                    int ii = stride * oi * RB_q;
                    int oj = oj_b * RB_p;
                    int oi = oi_b * RP_q;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            for (int k = 0; k < VLEN + 1; k++) {
                                for (int c = 0; c < VLEN + 1; c++) {
                                    for (int p = 0; p < RB_p + 1; p++) {
                                        for (int q = 0; q < RB_q + 1; q++) {
                                            int ijo = ij + stride * p;
                                            int iio = ii + stride * q;
                                              O[n][k][oj+p][oi+q][k] += W[k_b][c_b][r][s][c][k] ∗ I[n][c_b][ij0 + r][ii0 + s][c]
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }
    }
}

//Backward Propagation Loops w/ small GeMM calls
#pragma omp parallel for
for (int n = 0; n < N; n++) {
    for (int k_b = 0; k_b < K_b; k_b++) {
        for (int c_b = 0; c_b < C_b; c_b++) {
            for (int oj = 0; oj < P; oj ++) {
                ij = stride * oj;
                oi = 0;
                ii = 0;
                for (int r = 0; r < R; r++) {
                    for (int s = 0; s < S; s++) {
                        //I[n][c][ij+r][ii+s] += O[n][k][oj][oi] * W[k][c][r][s];
                        GEMM(&W[c_b][k_b][R - 1 - r][S - 1 -s][0][0], &dO[n][k_b][oj][oi][0], &dI[n][c_b][ij+r][ii+s][0]);
                    }
                }
            }
        }
    }
}

//Optimized Weight Update Gradient Loops
int P_b = P/B_p;
int Q_b = Q/B_q;

#pragma omp parallel for
for (int n = 0; n < N; n++) {
    for (int k_b = 0; k_b< K_b; k_b++) {
        for (int c_b = 0; c_b < C_b; c_b++) {
            for (int oj_b = 0; oj_b < P_b; oj_b++) {
                for (int oi_b = 0; oi_b < Q_b; oi_b++) {
                    int ij = stride * oj_b * B_p;
                    int ii = stride * oi_b * B_q;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            for (int p = 0; p < B_p + 1; p++) {
                                for (int q = 0; q < B_q + 1; q++) {
                                    for (int k = 0; k < VLEN + 1; k++) {
                                        for (int c = 0; c < VLEN + 1; c++) {
                                            ii += stride * p;
                                            ij += stride * q;
                                            //dW[k_b][c_b][r][s][c][k] += I[n][c_b][ij+r][ii+s][c] * dO[n][k_b][oj+p][oi+q][k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
