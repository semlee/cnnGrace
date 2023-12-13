#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>

//used for performance count
#include <chrono>
#include <ratio>
#include <cmath>

#if defined(_OPENMP)
# include <omp.h>
#endif
#include <arm_sve.h>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void arm_sve_conv_fp(conv_t* param, const float* input, float* output, const float* filter, const float* bias) {
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
    #pragma omp parallel for private(img, ofm_b, ifm_b, oj, oi, ij, ii, kj, ki, p, q, ijo, iio)
#endif                                              
    for (img = 0; img < nImg; img++) { //N
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) { //C_b
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) {  //K_b
                for (oj_b = 0; oj_b < ofh_b; oj_b++) { //P_b
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi_b < ofw_b; oi_b++) { //Q_b
                        oi = oi_b * RB_p;
                        ii = oi * stride_w;
                        for (kj = 0; kj < kh; kj++) { //R
                            for (ki = 0; ki < kw; ki++) { //S
                                for (p = 0; p < RB_p; p++) { //P
                                ijo = ij + stride_h * p - pad_h;
                                if (ijo + kj < 0 || ijo + kj >= ifh) continue;
                                for (q = 0; q < RB_q; q++) { //Q
                                    iio = ii + stride_w * q - pad_w;
                                    if (iio + ki < 0 || iio + ki >= ifw) continue;      
                                        size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                                ifm_b * ifhp * ifwp * VLEN+ 
                                                                (ijo + kj) * ifwp * VLEN + 
                                                                (iio + ki) * VLEN;
                                                                
                                        size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                                                                ofm_b * ofhp * ofwp * VLEN + 
                                                                (oj + p) * ofwp * VLEN + 
                                                                (oi + q) * VLEN;

                                        size_t filterIndex =    ofm_b * nIfm * kh * kw * VLEN + 
                                                                ifm_b * kh * kw * VLEN * VLEN + 
                                                                kj * kw * VLEN * VLEN + 
                                                                ki * VLEN * VLEN;

                                        // Load vectors using SVE intrinsics
                                        svfloat32_t inputVector = svld1_f32(svptrue_b32(), input + inputIndex);
                                        svfloat32_t filterVector = svld1_f32(svptrue_b32(), filter + filterIndex);
                                        svfloat32_t outputVector = svld1_f32(svptrue_b32(), output + outputIndex);

                                        // run Vector MAC Unit
                                        outputVector = svmla_f32_m(svptrue_b32(), outputVector, inputVector, filterVector);

                                        // Store result back
                                        svst1_f32(svptrue_b32(), output + outputIndex, outputVector);
                                    }
                                }
                            }
                        }
                    }
                }
            }
#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
        // Apply ReLU activation function
            for (int oj = 0; oj < ofh; oj++) {
                for (int oi = 0; oi < ofw; oi++) {
                    int reluIndex = img * nOfm_b * ofhp * ofwp +
                                    ofm_b * nOfm_b * ofhp * ofwp +
                                    oj * ofwp +
                                    oi;
                    output[reluIndex] = (output[reluIndex] < 0.0f) ? 0.0f : output[reluIndex];
                }
            }
#endif
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

void fill_random(float* input_array, size_t A = 1, size_t B = 1, size_t C = 1, size_t D = 1, size_t E = 1, size_t F = 1) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < A; i++) {
        for (size_t j = 0; j < B; j++) {
            for (size_t k = 0; k < C; k++) {
                for (size_t l = 0; l < D; l++) {
                    for (size_t m = 0; m < E; m++) {
                        for (size_t n = 0; n < F; n++) {
                            // Convert multi-dimensional indices to a flat index
                            size_t flatIndex = i * B * C * D * E * F +
                                               j * C * D * E * F + 
                                               k * D * E * F + 
                                               l * E * F +
                                               m * F + 
                                               n;
                            // Generate a random float value between -1 and 1
                            input_array[flatIndex] = dis(gen);
                        }
                    }
                    
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

    conv_t conv_param;

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

    // Additional Setting for vectorization
    /* initially using fixed dataset, will have argv to set VLEN */
    int VLEN = 4;
    int RB_p= 1;
    int RB_q = 1;

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
 
    /* set struct for register blocked convolution */
    conv_param.nImg = nImg;
    conv_param.nIfm = nIfm;
    conv_param.nOfm = nOfm;
    conv_param.ifhp = ifhp;
    conv_param.ifwp = ifwp;
    conv_param.ofhp = ofhp;
    conv_param.ofwp = ofwp;
    conv_param.ifh = ifh;
    conv_param.ifw = ifw;
    conv_param.ofh = ofh;
    conv_param.ofw = ofw;
    conv_param.pad_h = pad_h;
    conv_param.pad_w = pad_w;
    conv_param.pad_h_in = pad_h_in;
    conv_param.pad_w_in = pad_w_in;
    conv_param.pad_h_out = pad_h_out;
    conv_param.pad_w_out = pad_w_out;
    conv_param.kh = kh;
    conv_param.kw = kw;
    conv_param.stride_h = stride_h;
    conv_param.stride_w = stride_w;
    conv_param.VLEN = VLEN;
    conv_param.RB_p = RB_p;
    conv_param.RB_q = RB_q;

    // Calculate the total sizes
    size_t inputSize = nImg * nIfm * ifhp * ifwp;
    size_t outputSize = nImg * nOfm * ofhp * ofwp;
    size_t filterSize = nOfm * nIfm * kh * kw;

    
    // /* Allocate memory for naive arrays */
    // float* naive_input = new float[inputSize];
    // float* naive_input_save = new float[inputSize];

    // float* naive_output = new float[outputSize];
    // float* naive_output_save = new float[outputSize];
    // float* naive_output_bp = naive_output;
    // float* naive_output_wu = naive_output;

    // float* naive_filter = new float[filterSize];
    // float* naive_filter_save = new float[filterSize];
    // float* naive_filter_wu = naive_filter;

    // float* naive_bias = new float[nOfm];
    // float* naive_dbias = new float[nOfm];

    // fill_random(naive_input, nImg, nIfm, ifhp, ifwp);
    // fill_random(naive_filter, nOfm, nIfm, kh, kw);

    // //IMPORTANT MALLOC : copy data to save
    // for (size_t i = 0; i < inputSize; i++) {
    //     naive_input_save[i] = naive_input[i];
    // }
    // for (size_t i = 0; i < filterSize; i++) {
    //     naive_filter_save[i] = naive_filter[i];
    //     naive_filter_wu[i] = naive_filter[i];
    // }

    // /* Allocate memory for real convolutional arrays */
    // float* conv_input = new float[inputSize];
    // float* conv_input_save = new float[inputSize];

    // float* conv_output = new float[outputSize];
    // float* conv_output_save = new float[outputSize];
    // float* conv_output_bp = conv_output;
    // float* conv_output_wu = conv_output;

    // float* conv_filter = new float[filterSize];
    // float* conv_filter_save = new float[filterSize];
    // float* conv_filter_wu = new float[filterSize];

    // float* conv_bias = new float[nOfm];
    // float* conv_dbias = new float[nOfm];

    // //IMPORTANT MALLOC : copy data to save
    // for (size_t i = 0; i < inputSize; i++) {
    //     conv_input[i] = naive_input[i];
    //     conv_input_save[i] = naive_input[i];
    // }
    // for (size_t i = 0; i < filterSize; i++) {
    //     conv_filter[i] = naive_filter[i];
    //     conv_filter_save[i] = naive_filter[i];
    //     conv_filter_wu[i] = naive_filter[i];
    // }
    
    float* conv_input = new float[inputSize];
    float* conv_output = new float[outputSize];
    float* conv_filter = new float[filterSize];
    float* conv_bias = new float[nOfm];
    fill_random(conv_input, nImg, nIfm, ifhp, ifwp);
    fill_random(conv_filter, nOfm, nIfm, kh, kw);
    bool debug = true;


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

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    printf("##########################################\n");
    printf("#            Naive Computation           #\n");
    printf("##########################################\n");
    // naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
    // naive_conv_bp(&naive_param, naive_input, naive_output_bp, naive_filter, naive_input_save);
    // naive_conv_uw(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);

    printf("##########################################\n");
    printf("#           Performance Analysis         #\n");
    printf("##########################################\n");

    if (type == 'A' || type == 'F') {
        cout << "##########################################\n";
        cout << "               FORWARD PASS               \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        arm_sve_conv_fp(&conv_param, conv_input, conv_output, conv_filter, conv_bias);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
        double l_total = (double)duration_sec.count();
        

        double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

        printf("Total Time = %.5g\n", (double)l_total);
        printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
        printf("fp time = %.5g\n", ((double)(l_total/iters)));
        printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

        cout << "Input" << endl;
        for (int i = 0; i < 10; i++) {
            cout << conv_input[i] << " ";
        }
        cout << endl;

        cout << "Filter" << endl;
        for (int i = 0; i < 10; i++) {
            cout << conv_filter[i] << " ";
        }
        cout << endl;

        cout << "Output" << endl;
        for (int i = 0; i < outputSize; i++) {
            cout << conv_output[i] << " ";
        }
        
        // arm_sve_conv_fp_original(&conv_param, conv_input, conv_output_save, conv_filter, conv_bias);
        // for (int i = 0; i < outputSize; i++) {
        //     if (conv_output[i] != conv_output_save[i]) {
        //         error_count++;
        //     }
        // }
        // cout << "Error Count: " << error_count << "/" << outputSize << "\n";

    }
    /*
    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
        cout << "##########################################\n";
        cout << "               BACKWARD PASS              \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        arm_sve_conv_bp(&conv_param, conv_input, conv_output_bp, conv_filter, conv_input_save);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        cout << "Total time consumed: " << duration_sec.count() << "ms\n";
    }
    if (type == 'A' || type == 'U') {
        cout << "##########################################\n";
        cout << "               UPDATE WEIGHT              \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        arm_sve_conv_uw(&conv_param, conv_input_save, conv_output_wu, conv_filter_wu);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        cout << "Total time consumed: " << duration_sec.count() << "ms\n";
    }
    */

    /*
    printf("##########################################\n");
    printf("#           Correctness Checking         #\n");
    printf("##########################################\n");

    if (type == 'A' || type == 'F') {
        cout << "##########################################\n";
        cout << "               FORWARD PASS               \n";
        cout << "##########################################\n";
        int error_count = 0;

        for (int i = 0; i < outputSize; i++) {
            if (conv_output[i] != naive_output[i]) {
                error_count++;
            }
        }
        cout << "Error Count: " << error_count << "/" << outputSize << "\n";

        

    }
    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
        cout << "##########################################\n";
        cout << "               BACKWARD PASS              \n";
        cout << "##########################################\n";
        int error_count = 0;

        for (int i = 0; i < inputSize; i++) {
            if (conv_input[i] != naive_input[i]) {
                error_count++;
            }
        }
        cout << "Error Count: " << error_count << "/" << inputSize << "\n";
    }
    if (type == 'A' || type == 'U') {
        cout << "##########################################\n";
        cout << "               UPDATE WEIGHT              \n";
        cout << "##########################################\n";
        int error_count = 0;

        for (int i = 0; i < filterSize; i++) {
            if (conv_filter_wu[i] != naive_filter_wu[i]) {
                error_count++;
            }
        }
        cout << "Error Count: " << error_count << "/" << filterSize << "\n";
    }
    */
    
    printf("##########################################\n");
    printf("#           Cleaning Up data...          #\n");
    printf("##########################################\n");

    // //free allocated memory
    // delete[] naive_input;
    // delete[] naive_input_save;
    // delete[] naive_output;
    // delete[] naive_output_save;
    // delete[] naive_output_bp;
    // delete[] naive_output_wu;
    // delete[] naive_filter;
    // delete[] naive_filter_save;
    // delete[] naive_filter_wu;
    // delete[] naive_bias;
    // delete[] naive_dbias;

    delete[] conv_input;
    delete[] conv_output;
    delete[] conv_filter;
    delete[] conv_bias;

    printf("##########################################\n");
    printf("#                Complete.               #\n");
    printf("##########################################\n");
    return 0;
}
 