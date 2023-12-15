#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "regblock_ms2.h"

//additional header for parallelization
#if defined(_OPENMP)
# include <omp.h>
#endif
// #include <arm_sve.h>

void reg_block_conv_fp(conv_t* param, const std::vector<float>& input, std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& bias) {
    // Fetch data from param struct
    int nImg      = param->nImg;
    int nIfm      = param->nIfm;
    int nOfm      = param->nOfm;
    int ifhp      = param->ifhp;
    int ifwp      = param->ifwp;
    int ofhp      = param->ofhp;
    int ofwp      = param->ofwp;
    int ifh       = param->ifh;
    int ifw       = param->ifw;
    int ofh       = param->ofh;
    int ofw       = param->ofw;
    int pad_h     = param->pad_h;
    int pad_w     = param->pad_w;
    int pad_h_in  = param->pad_h_in;
    int pad_w_in  = param->pad_w_in;
    int pad_h_out = param->pad_h_out;
    int pad_w_out = param->pad_w_out;
    int kh        = param->kh;
    int kw        = param->kw;
    int stride_h  = param->stride_h;
    int stride_w  = param->stride_w;
    int VLEN      = param->VLEN;
    int RB_p      = param->RB_p;
    int RB_q      = param->RB_q;

    int nIfm_b = nIfm/VLEN;
    int nOfm_b = nOfm/VLEN;
    int ofh_b = ofh/RB_p;
    int ofw_b = ofw/RB_q;
    int img, ofm_b, ifm_b, oj_b, oj, ij, oi_b, oi, ii, kj, ki, ofm, ifm, p, q, ijo, iio;

#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm_b, ifm_b, oj_b, oi_b, ij, ii, kj, ki, ofm, ifm, p, q)
#endif

    for (img = 0; img < nImg; img++) { //N
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) { //K_b
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) { //C_b
                for (oj_b = 0; oj_b < ofh_b; oj_b++) { //P_b
                    oj = oj_b * RB_p;
                    ij = oj * stride_h - pad_h;
                    for (oi_b = 0; oi_b < ofw_b; oi_b++) { //Q_b
                        oi = oi_b * RB_p;
                        ii = oi * stride_w - pad_w;
                        for (kj = 0; kj < kh; kj++) { //R
                            for (ki = 0; ki < kw; ki++) { //S
                                for (ofm = 0; ofm < VLEN; ofm++) { //k
                                    for (ifm = 0; ifm < VLEN; ifm++) { //c
                                        size_t filterIndex =    ofm_b * nIfm * kh * kw * VLEN + 
                                                                ifm_b * kh * kw * VLEN * VLEN + 
                                                                kj * kw * VLEN * VLEN + 
                                                                ki * VLEN * VLEN + 
                                                                ifm * VLEN + 
                                                                ofm;

                                        for (p = 0; p < RB_p; p++) { //P
                                            ijo = ij + stride_h * p;
                                            if (ijo + kj < 0 || ijo + kj >= ifh) continue;
                                                              
                                            for (q = 0; q < RB_q; q++) { //Q
                                                iio = ii + stride_w * q;
                                                if (iio + ki < 0 || iio + ki >= ifw) continue;
                                                size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                                        ifm_b * ifhp * ifwp * VLEN+ 
                                                                        (ijo + kj) * ifwp * VLEN + 
                                                                        (iio + ki) * VLEN +
                                                                        ifm;
                                                size_t outputIndex =    img * nOfm * ofhp * ofwp * RB_p * RB_q +
                                                                        ofm_b * ofhp * ofwp * VLEN * RB_p * RB_q +
                                                                        oj * ofwp * VLEN * RB_q +
                                                                        oi * VLEN * RB_p +
                                                                        ofm * RB_p * RB_q +
                                                                        p * RB_q + 
                                                                        RB_q;
                                                
                                                output[outputIndex] += filter[filterIndex] * input[inputIndex];
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

}

void reg_block_conv_bp(conv_t* param, std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& naive_input_save) {

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
    int n, k_b, c_b, oj, oi, ii, ij, r, s;

#if defined (_OPENMP)
    #pragma omp parallel for private(n, k_b, c_b, oj, oi, ii, ij, r, s)
#endif
    for (n = 0; n < N; n++) {
        for (k_b = 0; k_b < K_b; k_b++) {
            for (c_b = 0; c_b < C_b; c_b++) {
                for (oj = 0; oj < P; oj++) {
                    for (oi = 0; oi < Q; oi++) {
                        ij = stride_h * oj;
                        ii = stride_w * oi;
                        for (r = 0; r < R; r++) {
                            for (s = 0; s < S; s++) {
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

void reg_block_conv_uw(conv_t* param, const std::vector<float>& input, const std::vector<float>& output, std::vector<float>& filter) {

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
    int n, k_b, c_b, oj_b, oi_b, r, s, p, q, k, c, oj, oi, ij, ii;

#if defined (_OPENMP)
    #pragma omp parallel for private(n, k_b, c_b, oj_b, oi_b, r, s, p, q, k, c, oj, oi, ij, ii)
#endif
    for (n = 0; n < N; n++) {
        for (k_b = 0; k_b < K_b; k_b++) {
            for (c_b = 0; c_b < C_b; c_b++) {
                for (oj_b = 0; oj_b < P_b; oj_b++) {
                    for (oi_b = 0; oi_b < Q_b; oi_b++) {
                        oj = oj_b * RB_p;
                        oi = oi_b * RB_q;
                        ij = stride_h * oj;
                        ii = stride_w * oi;
                        for (r = 0; r < R; r++) {
                            for (s = 0; s < S; s++) {
                                for (p = 0; p < RB_p + 1; p++) {
                                    for (q = 0; q < RB_q + 1; q++) {
                                        for (k = 0; k < VLEN + 1; k++) {
                                            for (c = 0; c < VLEN + 1; c++) {
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