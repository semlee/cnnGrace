#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "regblock_ms2.h"

//additional header for parallelization
#if defined(_OPENMP)
# include <omp.h>
#endif
// #include <arm_sve.h>

void CONV(const std::vector<float> input, std::vector<float> output, const std::vector<float> filter, int kh, int kw, int ifh, int ifw, int ofm_b, int ifm_b, int nOfm, int nIfm, int VLEN, int RB_p, int RB_q, int oj, int oi, int ij, int ii, int ifwp, int ofwp, int stride_h, int stride_w) {
   int kj, ki, ofm, ifm, p, q, ij0, ii0;
   for (kj = 0; kj < kh; ++kj) { //R
        if (ij+kj < 0 || ij+kj >= ifh) continue;
        for (ki = 0; ki < kw; ++ki) { //S
            if (ii+ki < 0 || ii+ki >= ifw) continue;
            for (ofm = 0; ofm < VLEN && ofm_b * VLEN + ofm < nOfm; ofm++) {
                for (ifm = 0; ifm < VLEN && ifm_b * VLEN + ifm < nIfm; ifm++) {
                    size_t filterIndex =    kj * kw * VLEN * VLEN + 
                                            ki * VLEN * VLEN + 
                                            ifm * VLEN + 
                                            ofm;
                    for (p = 0; p < RB_p; p++) {
                        ij0 = ij + stride_h * p;
                        if (ij0 + kj < 0 || ij0 + kj >= ifh) continue;   
                        for (q = 0; q < RB_q; q++) {
                            ii0 = ii + stride_w * q;
                            if (ii0 + ki < 0 || ii0 + ki >= ifw) continue; 
                            size_t inputIndex =     (ij0 + kj) * ifwp * VLEN + 
                                                    (ii0 + ki) * VLEN +
                                                    ifm;
                            size_t outputIndex =    (oj + p) * ofwp * VLEN + 
                                                    (oi + q) * VLEN +
                                                    ofm;
                            
                            output[outputIndex] += filter[filterIndex] * input[inputIndex];
                        }
                    }
                }
            }
        }
    }
}

void reg_block_conv_fp_mod(conv_t* param, const std::vector<float>& input, std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& bias) {
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

    int nIfm_b = nIfm / VLEN + (nIfm % VLEN != 0);
    int nOfm_b = nOfm / VLEN + (nOfm % VLEN != 0);
    int ofh_b = ofh/RB_p;
    int ofw_b = ofw/RB_q;
    int img, ofm_b, ifm_b, oj_b, oj, ij, oi_b, oi, ii, kj, ki, ofm, ifm, p, q, ij0, ii0;

    for (img = 0; img < nImg; ++img) { //N
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) { //K
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) { //C
                for (oj_b = 0; oj_b < ofh_b; ++oj_b) { //P
                    for (oi_b = 0; oi_b < ofw_b; ++oi_b) { //Q
                        oj = oj_b * RB_p;
                        ij = oj * stride_h - pad_h;
                        oi = oi_b * RB_q;
                        ii = oi * stride_w - pad_w;
                        auto inputIndex = input.begin() + img * nIfm_b * ifhp * ifwp * VLEN + ifm_b * ifhp * ifwp * VLEN;
                        auto outputIndex = output.begin() + img * nOfm_b * ofhp * ofwp * VLEN + ofm_b * ofhp * ofwp * VLEN;
                        auto filterIndex = filter.begin() + ofm_b * nIfm_b * kh * kw * VLEN * VLEN + ifm_b * kh * kw * VLEN * VLEN;
                        auto subvecSize = kh * kw * VLEN * VLEN * RB_p * RB_q;
                        CONV(std::vector<float>(inputIndex, inputIndex + subvecSize), 
                            std::vector<float>(outputIndex, outputIndex + subvecSize),
                            std::vector<float>(filterIndex, filterIndex + subvecSize),
                            kh, kw, ifh, ifw, ofm_b, ifm_b, nOfm, nIfm, VLEN, RB_p, RB_q, oj, oi, ij, ii, ifwp, ofwp, stride_h, stride_w);
                    }
                }
            }
        }
    }
    
}

void reg_block_conv_fp(conv_t* param, const std::vector<float>& input, std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& bias)  {
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

    int nIfm_b = nIfm / VLEN + (nIfm % VLEN != 0);
    int nOfm_b = nOfm / VLEN + (nOfm % VLEN != 0);
    int ofh_b = ofh/RB_p;
    int ofw_b = ofw/RB_q;
    int img, ofm_b, ifm_b, oj_b, oj, ij, oi_b, oi, ii, kj, ki, ofm, ifm, p, q, ij0, ii0;


    for (img = 0; img < nImg; ++img) { //N
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) { //K
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) { //C
                for (oj_b = 0; oj_b < ofh_b; ++oj_b) { //P
                    oj = oj_b * RB_p;
                    ij = oj * stride_h - pad_h;
                    for (oi_b = 0; oi_b < ofw_b; ++oi_b) { //Q
                        oi = oi_b * RB_q;
                        ii = oi * stride_w - pad_w;
                        for (kj = 0; kj < kh; ++kj) { //R
                            for (ki = 0; ki < kw; ++ki) { //S
                                for (ofm = 0; ofm < VLEN && ofm_b * VLEN + ofm < nOfm; ofm++) {
                                    for (ifm = 0; ifm < VLEN && ifm_b * VLEN + ifm < nIfm; ifm++) {
                                        size_t filterIndex =    ofm_b * nIfm_b * kh * kw * VLEN * VLEN + 
                                                                ifm_b * kh * kw * VLEN * VLEN + 
                                                                kj * kw * VLEN * VLEN + 
                                                                ki * VLEN * VLEN +
                                                                ifm * VLEN +
                                                                ofm;
                                        for (p = 0; p < RB_p; p++) {
                                            ij0 = ij + stride_h * p;
                                            if (ij0 + kj < 0 || ij0 + kj >= ifh) continue;   
                                            for (q = 0; q < RB_q; q++) {
                                                ii0 = ii + stride_w * q;
                                                if (ii0 + ki < 0 || ii0 + ki >= ifw) continue; 
                                                size_t inputIndex =     img * nIfm_b * ifhp * ifwp * VLEN + 
                                                                        ifm_b * ifhp * ifwp * VLEN + 
                                                                        (ij0 + kj) * ifwp * VLEN + 
                                                                        (ii0 + ki) * VLEN +
                                                                        ifm; 
                                                size_t outputIndex =    img * nOfm_b * ofhp * ofwp * VLEN + 
                                                                        ofm_b * ofhp * ofwp * VLEN+ 
                                                                        (oj + p) * ofwp * VLEN+ 
                                                                        (oi + q) * VLEN +
                                                                        ofm;

                                                output[outputIndex] += input[inputIndex] * filter[filterIndex];        
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

void reg_block_conv_fp_lanigiro(conv_t* param, const std::vector<float>& input, std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& bias)  {
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

    int nIfm_b = nIfm / VLEN + (nIfm % VLEN != 0);
    int nOfm_b = nOfm / VLEN + (nOfm % VLEN != 0);
    int ofh_b = ofh/RB_p;
    int ofw_b = ofw/RB_q;
    int img, ofm_b, ifm_b, oj_b, oj, ij, oi_b, oi, ii, kj, ki, ofm, ifm, p, q, ij0, ii0;


    for (img = 0; img < nImg; ++img) { //N
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) { //K
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) { //C
                for (oj_b = 0; oj_b < ofh_b; ++oj_b) { //P
                    oj = oj_b * RB_p;
                    ij = oj * stride_h - pad_h;
                    for (oi_b = 0; oi_b < ofw_b; ++oi_b) { //Q
                        oi = oi_b * RB_q;
                        ii = oi * stride_w - pad_w;
                        for (kj = 0; kj < kh; ++kj) { //R
                            for (ki = 0; ki < kw; ++ki) { //S
                                for (ofm = 0; ofm < VLEN && ofm_b * VLEN + ofm < nOfm; ofm++) {
                                    for (ifm = 0; ifm < VLEN && ifm_b * VLEN + ifm < nIfm; ifm++) {
                                        for (p = 0; p < RB_p; p++) {
                                            ij0 = ij + stride_h * p;
                                            if (ij0+kj < 0 || ij0+kj >= ifh) continue;
                                            for (q = 0; q < RB_q; q++) {
                                                ii0 = ii + stride_w * q;
                                                if (ii0+ki < 0 || ii0+ki >= ifw) continue;
                                                size_t inputIndex =     img * nIfm_b * ifhp * ifwp * VLEN + 
                                                                        ifm_b * ifhp * ifwp * VLEN + 
                                                                        ifm * ifhp * ifwp +
                                                                        (ij0 + kj) * ifwp + 
                                                                        (ii0 + ki);
                                                size_t outputIndex =    img * nOfm_b * ofhp * ofwp * VLEN + 
                                                                        ofm_b * ofhp * ofwp * VLEN + 
                                                                        ofm * ofhp * ofwp +
                                                                        (oj + p) * ofwp + 
                                                                        (oi + q);
                                                size_t filterIndex =    ofm_b * nIfm_b * kh * kw * VLEN * VLEN + 
                                                                        ofm * nIfm_b * kh * kw * VLEN +
                                                                        ifm_b * kh * kw * VLEN + 
                                                                        ifm * kh * kw +
                                                                        kj * kw + 
                                                                        ki;

                                                output[outputIndex] += input[inputIndex] * filter[filterIndex];        
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