#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "regblock_ms2.h"

//additional header for parallelization
#if defined(_OPENMP)
# include <omp.h>
#endif
// #include <arm_sve.h>

void arm_sve_conv_fp(conv_t* param, const float* input, float* output, const float* filter, const float* bias) {
    // Fetch data from param struct
    int N         = param->nImg;
    int C         = param->nIfm;
    int K         = param->nOfm;
    int ifhp      = param->ifhp;
    int ifwp      = param->ifwp;
    int ofhp      = param->ofhp;
    int ofwp      = param->ofwp;
    int ifh       = param->ifh;
    int ifw       = param->ifw;
    int P         = param->ofh;
    int Q         = param->ofw;
    int pad_h     = param->pad_h;
    int pad_w     = param->pad_w;
    int pad_h_in  = param->pad_h_in;
    int pad_w_in  = param->pad_w_in;
    int pad_h_out = param->pad_h_out;
    int pad_w_out = param->pad_w_out;
    int R         = param->kh;
    int S         = param->kw;
    int stride_h  = param->stride_h;
    int stride_w  = param->stride_w;
    int VLEN      = param->VLEN;
    int RB_p      = param->RB_p;
    int RB_q      = param->RB_q;

    int C_b = C/VLEN;
    int K_b = K/VLEN;
    int P_b = P/RB_p;
    int Q_b = Q/RB_q;

#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    for (int n = 0; n < N; n++) {
        for (int k_b = 0; k_b < K_b; k_b++) {
            for (int c_b = 0; c_b < C_b; c_b++) {
                for (int oj = 0; oj < P_b; oj++) {
                    int ij = oj * stride_h - pad_h;
                    for (int oi = 0; oi < Q_b; oi++) {
                        int ii = oi * stride_w - pad_w;
                        for (int r = 0; r < R; r++) {
                            if (ij + r < 0 || ij + r >= ifh) continue;
                            for (int s = 0; s < S; s++) {
                                if (ii + s < 0 || ii + s >= ifw) continue;

                                for (int c = 0; c <= VLEN; c++) {
                                    for (int k = 0; k <= VLEN; k++) {
                                        for (int p = 0; p <= RB_p; p++) {
                                            for (int q = 0; q <= RB_q; q++) {
                                                int ijo = ij + stride_h * p;
                                                int iio = ii + stride_w * q;
                                                //O[n][k_b][oj+p][oi+q][k] += W[k_b][c_b][r][s][c][k] ∗ I[n][c_b][ijo + r][iio + s][c]
                                                // Check boundary conditions
                                                // int inputIndex = (n * C_b * ((P_b * stride_h - pad_h) + stride_h * RB_p + R) * ((Q_b * stride_w - pad_h) + stride_h * RB_q + S) * VLEN) + 
                                                // (c_b * ((P_b * stride_h - pad_h) + stride_h * RB_p + R) * ((Q_b * stride_w - pad_h) + stride_h * RB_q + S) * VLEN) + 
                                                // ((ijo + r) * ((Q_b * stride_w - pad_h) + stride_h * RB_q + S) * VLEN) + ((iio + s) * VLEN) + c;
                                                // int outputIndex = (n * K_b * (P_b + RB_p) * (Q_b + RB_q) * VLEN) + (k_b * (P_b + RB_p) * (Q_b + RB_q) * VLEN) + ((oj + p) * (Q_b + RB_q) * VLEN) + ((oi + q) * VLEN) + k;
                                                // int filterIndex = (k_b * C_b * R * S * VLEN * VLEN) + (c_b * R * S * VLEN * VLEN) + (r * S * VLEN * VLEN) + (s * VLEN * VLEN) + (c * VLEN) + k;
                                                size_t inputIndex = n * C_b * ifhp * ifwp * VLEN +
                                                                    c_b * ifhp * ifwp * VLEN +
                                                                    (ijo + r) * ifwp * VLEN +
                                                                    (iio + s) * VLEN + c;
                                                
                                                size_t outputIndex = n * K_b * P_b * Q_b * VLEN +
                                                                    k_b * P_b * Q_b * VLEN +
                                                                    (oj + p) * Q_b * VLEN +
                                                                    (oi + q) * VLEN + k;

                                                size_t filterIndex = k_b * C_b * R * S * VLEN * VLEN +
                                                                    c_b * R * S * VLEN * VLEN +
                                                                    r * S * VLEN * VLEN +
                                                                    s * VLEN * VLEN +
                                                                    c * VLEN + k;
                                                output[outputIndex] += input[inputIndex] * filter[filterIndex];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
                    // Apply ReLU activation function
                    for (int oj = 0; oj < P_b * RB_p; oj++) {
                        for (int oi = 0; oi < Q_b * RB_q; oi++) {
                            int reluIndex = n * K_b * P_b * Q_b * VLEN * VLEN + k_b * P_b * Q_b * VLEN * VLEN + oj * Q_b * VLEN * VLEN + oi * VLEN * VLEN;
                            output[reluIndex] = (output[reluIndex] < 0.0f) ? 0.0f : output[reluIndex];
                        }
                    }
#endif

                }
            }
        }
    }

}

void arm_sve_conv_bp(conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save) {

    // Fetch data from param struct
    int N         = param->nImg;
    int C         = param->nIfm;
    int K         = param->nOfm;
    int ifhp      = param->ifhp;
    int ifwp      = param->ifwp;
    int ofhp      = param->ofhp;
    int ofwp      = param->ofwp;
    int ifh       = param->ifh;
    int ifw       = param->ifw;
    int P         = param->ofh;
    int Q         = param->ofw;
    int pad_h     = param->pad_h;
    int pad_w     = param->pad_w;
    int pad_h_in  = param->pad_h_in;
    int pad_w_in  = param->pad_w_in;
    int pad_h_out = param->pad_h_out;
    int pad_w_out = param->pad_w_out;
    int R         = param->kh;
    int S         = param->kw;
    int stride_h  = param->stride_h;
    int stride_w  = param->stride_w;
    int VLEN      = param->VLEN;

    int C_b = C/VLEN;
    int K_b = K/VLEN;

#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    for (int n = 0; n < N; n++) {
        for (int k_b = 0; k_b < K_b; k_b++) {
            for (int c_b = 0; c_b < C_b; c_b++) {
                for (int oj = 0; oj < P; oj++) {
                    for (int oi = 0; oi < Q; oi++) {
                        int ij = stride_h * oj;
                        int ii = stride_w * oi;
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                // Compute flat indices
                                size_t inputIndex = (n * C_b *(P + R) * (Q + S)) + (c_b * (P + R) * (Q + S)) + ((ij + r) * (Q + S)) + (ii + s);
                                size_t outputIndex = (n * K_b * P * Q) + (k_b * P * Q) + (oj * Q) + oi;
                                size_t filterIndex = (c_b * K_b * R * S) + (k_b * R * S) + (R - 1 - r) * S + (S - 1 - s);
                                // GEMM(&W[c_b][k_b][R - 1 - r][S - 1 -s][0][0], &dO[n][k_b][oj][oi][0], &dI[n][c_b][ij+r][ii+s][0]);
                                // Perform the convolution
                                input[inputIndex] += output[outputIndex] * filter[filterIndex];
                            }
                        }
                    }
                }

#if defined(USE_FUSED_RELU_BWD)
                for (int ij = 0; ij < ifh; ij++) {
                    for (int ii = 0; ii < ifw; ii++) {
                        if (naive_input_save[n * nIfm * ifh * ifw + c_b * ifh * ifw + ij * ifw + ii] == 0.0) {
                            input[n * nIfm * ifh * ifw + c_b * ifh * ifw + ij * ifw + ii] = 0.0;
                        }
                    }
                }
#endif
            }
        }
    }

}

void arm_sve_conv_uw(conv_t* param, const float* input, const float* output, float* filter) {

    // Fetch data from param struct
    int N         = param->nImg;
    int C         = param->nIfm;
    int K         = param->nOfm;
    int ifhp      = param->ifhp;
    int ifwp      = param->ifwp;
    int ofhp      = param->ofhp;
    int ofwp      = param->ofwp;
    int ifh       = param->ifh;
    int ifw       = param->ifw;
    int P         = param->ofh;
    int Q         = param->ofw;
    int pad_h     = param->pad_h;
    int pad_w     = param->pad_w;
    int pad_h_in  = param->pad_h_in;
    int pad_w_in  = param->pad_w_in;
    int pad_h_out = param->pad_h_out;
    int pad_w_out = param->pad_w_out;
    int R         = param->kh;
    int S         = param->kw;
    int stride_h  = param->stride_h;
    int stride_w  = param->stride_w;
    int VLEN      = param->VLEN;
    int RB_p      = param->RB_p;
    int RB_q      = param->RB_q;

    int C_b = C/VLEN;
    int K_b = K/VLEN;
    int P_b = P/RB_p;
    int Q_b = Q/RB_q;


#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    for (int n = 0; n < N; n++) {
        for (int k_b = 0; k_b < K_b; k_b++) {
            for (int c_b = 0; c_b < C_b; c_b++) {
                for (int oj_b = 0; oj_b < P_b; oj_b++) {
                    for (int oi_b = 0; oi_b < Q_b; oi_b++) {
                        int oj = oj_b * RB_p;
                        int oi = oi_b * RB_q;
                        int ij = stride_h * oj;
                        int ii = stride_w * oi;
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                for (int p = 0; p < RB_p + 1; p++) {
                                    for (int q = 0; q < RB_q + 1; q++) {
                                        for (int k = 0; k < VLEN + 1; k++) {
                                            for (int c = 0; c < VLEN + 1; c++) {
                                                ij += stride_h * p;
                                                ii += stride_w * q;
                                                
                                                // Compute flat indices
                                                size_t inputIndex = (n * C_b * (P_b * RB_p + R) * (Q_b * RB_q + S) * VLEN) + (c_b * (P_b * RB_p + R) * (Q_b * RB_q + S) * VLEN) + ((ij + r) * (Q_b * RB_q + S) * VLEN) + ((ii + s) * VLEN) + c;
                                                size_t outputIndex = (n * K_b * (P_b * RB_p + RB_p) * (Q_b * RB_q + RB_q) * VLEN) + (k_b * (P_b * RB_p + RB_p) * (Q_b * RB_q + RB_q) * VLEN) + ((oj + p) * (Q_b * RB_q + RB_q) * VLEN) + ((oi + q) * VLEN) + k;
                                                size_t filterIndex = (k_b * C_b * R * S * VLEN * VLEN) + (c_b * R * S * VLEN * VLEN) + (r * S * VLEN * VLEN) + (s * VLEN * VLEN) + (c * VLEN) + k;
                                                //dW[k_b][c_b][r][s][c][k] += I[n][c_b][ij+r][ii+s][c] * dO[n][k_b][oj+p][oi+q][k];
                                                // Perform the convolution
                                                filter[filterIndex] += input[inputIndex] * output[outputIndex];
                                            }
                                        }
                                        ii += stride_w * q;
                                    }
                                    ij += stride_h * p;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

} 