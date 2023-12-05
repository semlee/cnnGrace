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
                        for (int kj = 0; kj < R; kj++) {
                            if (ij + kj < 0 || ij + kj >= ifh) continue;
                            for (int ki = 0; ki < S; ki++) {
                                if (ii + ki < 0 || ii + ki >= ifw) continue;

                                for (int c = 0; c < VLEN; c++) {
                                    for (int k = 0; k < VLEN; k++) {
                                        for (int p = 0; p < RB_p; p++) {
                                            for (int q = 0; q < RB_q; q++) {
                                                int ijo = ij + stride_h * p;
                                                int iio = ii + stride_w * q;

                                                // Check boundary conditions
                                                if (ijo >= 0 && ijo < ifh && iio >= 0 && iio < ifw) {
                                                    int inputIndex = n * C_b * ifh * ifw + (c_b * VLEN + c) * ifh * ifw + (ijo + kj) * ifw + (iio + ki);
                                                    int outputIndex = n * K_b * P_b * Q_b * VLEN * VLEN + k_b * P_b * Q_b * VLEN * VLEN + oj * Q_b * VLEN * VLEN + oi * VLEN * VLEN + c * VLEN + k;
                                                    int filterIndex = k_b * C_b * R * S * VLEN * VLEN + c_b * R * S * VLEN * VLEN + kj * S * VLEN * VLEN + ki * VLEN * VLEN + c * VLEN + k;

                                                    output[outputIndex] += input[inputIndex] * filter[filterIndex];
                                                }
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
                    int ij = stride_h * oj;
                    int oi = 0;
                    int ii = 0;
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            // Compute flat indices
                            size_t inputIndex = n * C_b * ifh * ifw + c_b * ifh * ifw + (ij + r) * ifw + (ii + s);
                            size_t outputIndex = n * K_b * P * Q + k_b * P * Q + oj * Q + oi;
                            size_t filterIndex = c_b * K_b * R * S + k_b * R * S + r * S + s;

                            // Perform the convolution
                            input[inputIndex] += output[outputIndex] * filter[filterIndex];
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
                    int ij = stride_h * oj_b * RB_p;
                    int ii = 0;  // Reset ii for each oj_b
                    for (int oi_b = 0; oi_b < Q_b; oi_b++) {
                        int oi = oi_b * RB_q;
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                for (int p = 0; p < RB_p + 1; p++) {
                                    for (int q = 0; q < RB_q + 1; q++) {
                                        for (int k = 0; k < VLEN + 1; k++) {
                                            for (int c = 0; c < VLEN + 1; c++) {
                                                // Compute flat indices
                                                size_t inputIndex = n * C_b * ifh * ifw + c_b * ifh * ifw + (ij + r) * ifw + (ii + s);
                                                size_t outputIndex = n * K_b * P_b * Q_b * VLEN * VLEN * R * S + k_b * P_b * Q_b * VLEN * VLEN * R * S
                                                                    + oj_b * Q_b * VLEN * VLEN * R * S + oi_b * VLEN * VLEN * R * S
                                                                    + r * VLEN * VLEN * R * S + s * VLEN * R * S + q * VLEN * S + p * VLEN + c;
                                                size_t filterIndex = c_b * K_b * R * S * VLEN * VLEN + k_b * R * S * VLEN * VLEN + r * S * VLEN * VLEN + s * VLEN * VLEN
                                                                    + c * VLEN + k;

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