#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

//used for data generation (random)
#include <random>

//used for performance count
#include <chrono>
#include <ratio>
#include <cmath>

#if defined(_OPENMP)
# include <omp.h>
#endif
// #include <arm_sve.h>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

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
  int VLEN;
  int RB_p;
  int RB_q;
} conv_t;

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
    #pragma omp parallel for private(img, ofm_b, ifm_b, oj_b, oi_b, ij, ii, kj, ki, ofm, ifm, p, q)
#endif

    for (img = 0; img < nImg; img++) {
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) {
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) {
                for (oj_b = 0; oj_b < ofh_b; oj_b++) {
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi_b < ofw_b; oi_b++) {
                        oi = oi_b * RB_p;
                        ii = oi * stride_h;
                        for (kj = 0; kj < kh; kj++) {
                            if (ij + kj < 0 || ij + kj >= ifh) continue;
                            for (ki = 0; ki < kw; ki++) {
                                if (ii + ki < 0 || ii + ki >= ifw) continue;
                                for (ofm = 0; ofm < VLEN; ofm++) {
                                    for (ifm = 0; ifm < VLEN; ifm++) {
                                        for (p = 0; p < RB_p; p++) {
                                            for (q = 0; q < RB_q; q++) {
                                                ijo = ij + stride_h * p - pad_h;
                                                iio = ii + stride_w * p - pad_w;
                                                size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                                        ifm_b * ifhp * ifwp * VLEN+ 
                                                                        (ijo + kj) * ifwp * VLEN + 
                                                                        (iio + ki) * VLEN +
                                                                        ifm;
                                                size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                                                                        ofm_b * ofhp * ofwp * VLEN + 
                                                                        oj * ofwp * VLEN + 
                                                                        oi * VLEN +
                                                                        ofm;
                                                size_t filterIndex =    ofm_b * nIfm * kh * kw * VLEN * VLEN + 
                                                                        ifm_b * kh * kw * VLEN * VLEN + 
                                                                        kj * kw * VLEN * VLEN + 
                                                                        ki * VLEN * VLEN + 
                                                                        ofm * VLEN + 
                                                                        ifm;
                                                // size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                //                         (ifm_b * VLEN + ifm) * ifhp * ifwp + 
                                                //                         (ij + kj) * ifwp + 
                                                //                         (ii + ki);
                                                // size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                                                //                         (ofm_b * VLEN + ofm) * ofhp * ofwp + 
                                                //                         oj * ofwp + 
                                                //                         oi;
                                                // size_t filterIndex =    (ofm_b * VLEN + ofm) * nIfm * kh * kw + 
                                                //                         (ifm_b * VLEN + ifm) * kh * kw + 
                                                //                         kj * kw + 
                                                //                         ki;
                                                
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

void fill_random(float* input_array, size_t A = 1, size_t B = 1, size_t C = 1, size_t D = 1) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < A; i++) {
        for (size_t j = 0; j < B; j++) {
            for (size_t k = 0; k < C; k++) {
                for (size_t l = 0; l < D; l++) {
                    // Convert multi-dimensional indices to a flat index
                    size_t flatIndex = i * B * C * D + j * C * D + k * D + l;
                    // Generate a random float value between -1 and 1
                    input_array[flatIndex] = dis(gen);
                }
            }
        }
    }
}


int main (int argc, char** argv) {

    conv_t conv_param;

    volatile int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
    volatile int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
    
    //void* scratch;
    //size_t scratch_size = 0;

    // Data fetched from original layer_example_f32
    /* some parameters we can overwrite via cli,
        default is some inner layer of overfeat */
    volatile int iters = 10;         /* repetitions of benchmark */
    volatile int ifw = 14;           /* input width, "W" */
    volatile int ifh = 20;           /* input height, "H" */
    volatile int nImg = 32;          /* mini-batch size, "N" */
    volatile int nIfm = 256;         /* number of input feature maps, "C" */
    volatile int nOfm = 512;         /* number of output feature maps, "K" */
    volatile int kh = 3;             /* filter height, "R" */
    volatile int kw = 3;             /* filter width, "S" */
    volatile int padh = 0;           /* padding in input, height */
    volatile int padw = 0;           /* padding in input, width */
    volatile int stride = 1;         /* stride when accessing inputs */
    volatile int padding_mode = 0;   /* padding mode */
    volatile char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
    volatile char format = 'A';      /* 'A': ALL, 'L': LIBXSMM, 'T': Tensorflow, 'M', Mixed */

    // Additional Setting for vectorization
    /* initially using fixed dataset, will have argv to set VLEN */
    volatile int VLEN = 4;
    volatile int RB_p= 4;
    volatile int RB_q = 4;

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
    size_t inputSize = nImg * nIfm * ifhp * ifwp * VLEN;
    size_t outputSize = nImg * nOfm * ofhp * ofwp * VLEN;
    size_t filterSize = nOfm * nIfm * kh * kw * VLEN * VLEN;

    float* conv_input = new float[inputSize];
    float* conv_output = new float[outputSize];
    float* conv_filter = new float[filterSize];
    float* conv_bias = new float[nOfm];
    fill_random(conv_input, nImg, nIfm, ifhp, ifwp);
    fill_random(conv_filter, nOfm, nIfm, kh, kw);
    
    if (!conv_input || !conv_output || !conv_filter || !conv_bias) {
        // Handle memory allocation failure
        // You may want to throw an exception, log an error, or exit the program
        delete[] conv_input;
        delete[] conv_output;
        delete[] conv_filter;
        delete[] conv_bias;
        throw std::bad_alloc();
    }

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
        for (int i = 0; i < 10; i++) {
            cout << conv_output[i] << " ";
        }
        cout << endl;
    }
    printf("##########################################\n");
    printf("#           Cleaning Up data...          #\n");
    printf("##########################################\n");
    
    delete[] conv_input;
    delete[] conv_output;
    delete[] conv_filter;
    delete[] conv_bias;
    printf("##########################################\n");
    printf("#                Complete.               #\n");
    printf("##########################################\n");
    return 0;
}
 