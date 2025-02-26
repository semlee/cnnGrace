#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
//used for performance count
#include <chrono>
#include <ratio>
#include <cmath>

//additional header for parallelization
#include <omp.h>
// #include <arm_sve.h>


using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


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

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

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
                                // size_t inputIndex = img * nIfm * ifh * ifw + ifm * ifh * ifw + (ij + kj) * ifw + (ii + ki);
                                // size_t outputIndex = img * nOfm * ofh * ofw + ofm * ofh * ofw + oj * ofw + oi;
                                // size_t filterIndex = ofm * nIfm * kh * kw + ifm * kh * kw + kj * kw + ki;
                                size_t inputIndex = (img * nIfm * (ofh + kh) * (ofw + kw)) + (ifm * (ofh + kh) * (ofw + kw)) + ((ij + kj) * (ofw + kw)) + (ii + ki);
                                size_t outputIndex = (img * nOfm * ofh * ofw) + (ofm * ofh * ofw) + (oj * ofw) + oi;
                                size_t filterIndex = (ofm * nIfm * kh * kw) + (ifm * kh * kw) + (kj * kw) + ki;

                                // size_t inputIndex = img * nIfm * ifhp * ifwp + 
                                //                     ifm * ifhp * ifwp + 
                                //                     (ij + kj) * ifwp + 
                                //                     (ii + ki);
                                // size_t inputIndex = (img * nIfm * ((ofh * stride_h - pad_h) + kh) * ((ofw * stride_w - pad_w)+ kw)) + 
                                //                     (ifm * ((ofh * stride_h - pad_h) + kh) * ((ofw * stride_w - pad_w)+ kw)) + 
                                //                     ((ij + kj) * ((ofw * stride_w - pad_w)+ kw)) + 
                                //                     (ii + ki);
                                
                                output[outputIndex] += input[inputIndex] * filter[filterIndex];

                            }
                        }
                    }
                }
            }
        }
    }
}

void naive_conv_fp_original(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias) {
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
                                // size_t inputIndex = img * nIfm * ifh * ifw + ifm * ifh * ifw + (ij + kj) * ifw + (ii + ki);
                                // size_t outputIndex = img * nOfm * ofh * ofw + ofm * ofh * ofw + oj * ofw + oi;
                                // size_t filterIndex = ofm * nIfm * kh * kw + ifm * kh * kw + kj * kw + ki;

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

void fill_random(float* input_array, size_t A = 1, size_t B = 1, size_t C = 1, size_t D = 1) {
    // Seed the random number generator
    time_t t;
    srand(static_cast<unsigned int>(time(&t)));

    for (size_t i = 0; i < A; i++) {
        for (size_t j = 0; j < B; j++) {
            for (size_t k = 0; k < C; k++) {
                for (size_t l = 0; l < D; l++) {
                    // Convert multi-dimensional indices to a flat index
                    size_t flatIndex = i * B * C * D + j * C * D + k * D + l;
                    // Generate a random float value between 0 and 1
                    float random_value = static_cast<float>(rand()) / RAND_MAX;
                    // Round to the thousandth place
                    input_array[flatIndex] = round(random_value * 1000) / 1000.0f;
                }
            }
        }
    }
}

void fill_random_array(float* input_array, size_t indexSize) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < indexSize; i++) {
        input_array[i] = dis(gen);
    }
}

int main (int argc, char** argv) {

    // float *naive_input, *naive_output, *naive_output_save, *naive_filter, *naive_filter_wu, *naive_output_bp, *naive_output_wu, *naive_libxsmm_output;
    // float *naive_libxsmm_input, *naive_libxsmm_filter, *naive_input_save, *naive_filter_save, *naive_filter_kcrs;
    //float *input_nhwc, *output_nhwc, *filter_rsck, *dinput_nhwc, *doutput_nhwc, *dfilter_rsck, *naive_output_nhwc, *naive_input_nhwc;
    //float *naive_bias, *bias_libxsmm, *naive_dbias, *dbias_libxsmm, *bias_nhwc, *dbias_nhwc;
    //float *input_libxsmm, *filter_libxsmm, *output_libxsmm, *dinput_libxsmm, *dfilter_libxsmm, *doutput_libxsmm, *filtertr_libxsmm;
    //float *batchstats_libxsmm;

    naive_conv_t naive_param;

    int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
    int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
    
    //void* scratch;
    //size_t scratch_size = 0;

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

#if defined(_OPENMP)
    int nThreads = omp_get_max_threads(); /* number of threads */
#else
    int nThreads = 1; /* number of threads */
#endif

    /* reading new values from cli */
    int i = 1;
    if (argc > i) iters      = atoi(argv[i++]);
    if (argc > i) ifw        = atoi(argv[i++]);
    if (argc > i) ifh        = atoi(argv[i++]);
    if (argc > i) nImg       = atoi(argv[i++]);
    if (argc > i) nIfm       = atoi(argv[i++]);
    if (argc > i) nOfm       = atoi(argv[i++]);
    if (argc > i) kw         = atoi(argv[i++]);
    if (argc > i) kh         = atoi(argv[i++]);
    if (argc > i) padw       = atoi(argv[i++]);
    if (argc > i) padh       = atoi(argv[i++]);
    if (argc > i) stride     = atoi(argv[i++]);
    if (argc > i) type       = *(argv[i++]);
    if (argc > i) format     = *(argv[i++]);
    if (argc > i) padding_mode = atoi(argv[i++]);

    if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
        printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
        return 0;
    }


    stride_w = stride;
    stride_h = stride;
    pad_w = padw;
    pad_h = padh;

    if (0 == padding_mode) {
        pad_h_in = 0;
        pad_w_in = 0;
        pad_h_out = 0;
        pad_w_out = 0;
    }
    else {
        /* TODO: change "1" to "0" if "padding_mode = -1" is acknowledged */
        if (1 < padding_mode) pad_w = padding_mode;
        pad_h_in = pad_h;
        pad_w_in = pad_w;
        pad_h_out = pad_h;
        pad_w_out = pad_w;
    }

    /* deriving some values for naive code */
    ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
    ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
    ifhp = ifh + 2 * pad_h_in;
    ifwp = ifw + 2 * pad_w_in;
    ofhp = ofh + 2 * pad_h_out;
    ofwp = ofw + 2 * pad_w_out;
                                
    /* set struct for naive convolution */
    naive_param.nImg = nImg;
    naive_param.nIfm = nIfm;
    naive_param.nOfm = nOfm;
    naive_param.ifhp = ifhp;
    naive_param.ifwp = ifwp;
    naive_param.ofhp = ofhp;
    naive_param.ofwp = ofwp;
    naive_param.ifh = ifh;
    naive_param.ifw = ifw;
    naive_param.ofh = ofh;
    naive_param.ofw = ofw;
    naive_param.pad_h = pad_h;
    naive_param.pad_w = pad_w;
    naive_param.pad_h_in = pad_h_in;
    naive_param.pad_w_in = pad_w_in;
    naive_param.pad_h_out = pad_h_out;
    naive_param.pad_w_out = pad_w_out;
    naive_param.kh = kh;
    naive_param.kw = kw;
    naive_param.stride_h = stride_h;
    naive_param.stride_w = stride_w;
    /*
    naive_input           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_input_save      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_save     = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_bp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_wu       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_libxsmm_output  = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_libxsmm_input   = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_filter          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_save     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_wu       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_kcrs     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_libxsmm_filter  = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_bias            = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
    naive_dbias           = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
    
    svuint32_t naive_input           [nImg][nIfm][ifhp][ifwp];
    svuint32_t naive_input_save      [nImg][nIfm][ifhp][ifwp];
    svuint32_t naive_output          [nImg][nOfm][ofhp][ofwp];
    svuint32_t naive_output_save     [nImg][nOfm][ofhp][ofwp];
    svuint32_t naive_output_bp       [nImg][nOfm][ofhp][ofwp];
    svuint32_t naive_output_wu       [nImg][nOfm][ofhp][ofwp];
    svuint32_t naive_filter          [nOfm][nIfm][kh][kw];
    svuint32_t naive_filter_save     [nOfm][nIfm][kh][kw];
    svuint32_t naive_filter_wu       [nOfm][nIfm][kh][kw];
    svuint32_t naive_bias            [nOfm];
    svuint32_t naive_dbias           [nOfm];

    naive_input           = (float*)malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_input_save      = (float*)malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_output          = (float*)malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_save     = (float*)malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_bp       = (float*)malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_output_wu       = (float*)malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_libxsmm_output  = (float*)malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    naive_libxsmm_input   = (float*)malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    naive_filter          = (float*)malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_save     = (float*)malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_wu       = (float*)malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_filter_kcrs     = (float*)malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_libxsmm_filter  = (float*)malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
    naive_bias            = (float*)malloc( nOfm*               sizeof(float), 2097152);
    naive_dbias           = (float*)malloc( nOfm*               sizeof(float), 2097152);

    */
    
    // Calculate the total sizes
    size_t inputSize = nImg * nIfm * ifhp * ifwp;
    size_t outputSize = nImg * nOfm * ofhp * ofwp;
    size_t filterSize = nOfm * nIfm * kh * kw;

    // Allocate memory for the arrays
    float* naive_input = new float[inputSize];
    float* naive_output = new float[outputSize];
    float* naive_filter = new float[filterSize];
    float* naive_bias = new float[nOfm];
    fill_random_array(naive_input, inputSize);
    fill_random_array(naive_filter, filterSize);

    /* print some summary */
    printf("##########################################\n");
    printf("#          Setting Up (Common)           #\n");
    printf("##########################################\n");
    printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);

    printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
    printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
    printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );

#if defined(_OPENMP)
    double omp_start;
    double omp_end;
#endif
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    printf("##########################################\n");
    printf("#           Performance Analysis         #\n");
    printf("##########################################\n");

    ///*
    cout << "##########################################\n";
    cout << "               FORWARD PASS               \n";
    cout << "##########################################\n";

#if defined(_OPENMP)
        omp_start = omp_get_wtime();
#endif    
    start = high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
            naive_conv_fp_original(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
        }
    }
    end = high_resolution_clock::now();
#if defined(_OPENMP)
        omp_end = omp_get_wtime();
        double omp_time = (omp_end - omp_start);
#endif 
    duration_sec = std::chrono::duration_cast<duration<double, std::micro>>(end - start);
    //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
    double l_total = duration_sec.count() * 1e-6;
    

    double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;
#if defined(_OPENMP)
    printf("Openmp Time = %.5g\n", omp_time);
#endif  
    printf("Total Time = %.5g\n", (double)l_total);
    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,FP,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g\n", 
                nThreads, nImg, nIfm, nOfm, ifw, ifh, kw, kh, stride, padw, padh, 
                l_total, ((double)(l_total/iters)), (flops*1e-9)/l_total);

    // naive_conv_fp(&naive_param, naive_input, naive_output_save, naive_filter, naive_bias);
    // int error_count = 0;

    // for (int i = 0; i < outputSize; i++) {
    //     if (naive_output_save[i] != naive_output[i]) {
    //         error_count++;
    //     }
    // }
    // cout << "Error Count: " << error_count << "/" << outputSize << "\n";
    /*
    for (int i = 0; i < outputSize; i++) {
        cout << naive_output[i] << endl;
    }
    cout << endl;
    cout << endl;
    for (int i = 0; i < outputSize; i++) {
        cout << naive_output_save[i] << endl;
    }
    cout << endl;
    */
    //*/
    /*
    cout << "##########################################\n";
    cout << "               BACKWARD PASS              \n";
    cout << "##########################################\n";

    start = high_resolution_clock::now();
    naive_conv_bp(&naive_param, naive_input, naive_output_bp, naive_filter, naive_input_save);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "Total time consumed: " << duration_sec.count() << "ms\n";
     */
    /*
    cout << "##########################################\n";
    cout << "               UPDATE WEIGHT              \n";
    cout << "##########################################\n";

    start = high_resolution_clock::now();
    naive_conv_wu(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "Total time consumed: " << duration_sec.count() << "ms\n";
    */
    printf("##########################################\n");
    printf("#           Cleaning Up data...          #\n");
    printf("##########################################\n");

    //free allocated memory
    delete[] naive_input;
    delete[] naive_output;
    delete[] naive_filter;
    delete[] naive_bias;
    
    printf("##########################################\n");
    printf("#                Complete.               #\n");
    printf("##########################################\n");
    return 0;
}

