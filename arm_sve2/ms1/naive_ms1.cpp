#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "naive_ms1.h"

//additional header for parallelization
#include <omp.h>
// #include <arm_sve.h>

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

/*
N = nImg
K = nOfm
C = nIfm
P = ofh
Q = ofw
R = kh
S = kw

n = img
ofm = k
ifm = c
oj = oj
oi = oi
ii = ii
ij = ij
kj = r
ki = s
*/

void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias) {
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
    /* loop counters */
    int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

#if defined (_OPENMP)
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
                                // LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                                // LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                                // * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
                                // output[n][k][oj][oi] += input[n][c][ij + r][ii + s] ∗ filter[k][c][r][s];
                                // output[img][ofm][oj][oi] += input[img][ifm][ij + kj][ii + ki] ∗ filter[ofm][ifm][kj][ki];
                                
                                size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                        ifm * ifhp * ifwp + 
                                                        (ij + kj) * ifwp + 
                                                        (ii + ki);
                                size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                                                        ofm * ofhp * ofwp + 
                                                        oj * ofwp + 
                                                        oi;
                                size_t filterIndex =    ofm * nIfm * kh * kw + 
                                                        ifm * kh * kw + 
                                                        kj * kw + 
                                                        ki;

                                std::cout << outputIndex << " ";

                                output[outputIndex] += input[inputIndex] * filter[filterIndex];

                            }
                        }
                    }
                }
            }
        }
    }
}

void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save) {

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
    /* loop counters */
    int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    //Naive Backward Propagation Loops  
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
                                //dI[n][c][ij+r][ii+s] += dO[n][k][oj][oi] * W[k][c][r][s];
                                //input[img][ifm][ij+kj][ii+ki] += output[img][ofm][oj][oi] ∗ filter[ofm][ifm][kj][ki];
                                // Compute flat indices
                                size_t inputIndex = img * nIfm * ifh * ifw + ifm * ifh * ifw + (ij + kj) * ifw + (ii + ki);
                                size_t outputIndex = img * nOfm * ofh * ofw + ofm * ofh * ofw + oj * ofw + oi;
                                size_t filterIndex = ofm * nIfm * kh * kw + ifm * kh * kw + kj * kw + ki;

                                // Perform the convolution
                                input[inputIndex] += output[outputIndex] * filter[filterIndex];
                            }
                        }
                    }
                }
            }
        }
    }
}

void naive_conv_uw(naive_conv_t* param, const float* input, const float* output, float* filter) {

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
    /* loop counters */
    int img, ofm, ifm, oj, oi, ij, ii, kj, ki;


#if defined (_OPENMP)
    #pragma omp parallel for private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    //Naive Weight Update Gradient Loops
    for (img = 0; img < nImg; ++img) {
        for (ofm = 0; ofm < nOfm; ++ofm) {
            for (ifm = 0; ifm < nIfm; ++ifm) {
                for (oj = 0; oj < ofh; ++oj) {
                    ij = oj * stride_h;
                    for (oi = 0; oi < ofw; ++oi) {
                        ii = oi * stride_w;
                        for (kj = 0; kj < kh; ++kj) {
                            if (ij+kj < 0 || ij+kj >= ifh) continue;
                            for (ki = 0; ki < kw; ++ki) {
                                if (ii+ki < 0 || ii+ki >= ifw) continue;
                                // dW[k][c][r][s] += I[n][c][ij+r][ii+s] * dO[n][k][oj][oi];
                                //filter[ofm][ifm][kj][ki] += input[img][ifm][ij + kj][ii + ki] * output[img][ofm][oj][oi];
                                size_t inputIndex = img * nIfm * ifh * ifw + ifm * ifh * ifw + (ij + kj) * ifw + (ii + ki);
                                size_t outputIndex = img * nOfm * ofh * ofw + ofm * ofh * ofw + oj * ofw + oi;
                                size_t filterIndex = ofm * nIfm * kh * kw + ifm * kh * kw + kj * kw + ki;
                                filter[filterIndex] += input[inputIndex] * output[outputIndex];
                            }
                        }
                    }
                }
            }
        }
    } 
} 