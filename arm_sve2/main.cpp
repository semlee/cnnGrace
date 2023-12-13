#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <vector>

//include cumstom header files
#include "ms1/naive_ms1.h"
#include "ms2/regblock_ms2.h"
// #include "ms3/regsve_ms3.h"

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

void fill_random_array(std::vector<float>& input_array, size_t indexSize) {
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
    if (argc > i) VLEN       = atoi(argv[i++]);
    if (argc > i) RB_p       = atoi(argv[i++]);
    if (argc > i) RB_q       = atoi(argv[i++]);
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

    // /* Allocate memory for naive arrays */
    // /* NAIVE MEMORY */
    // // Forward Pass (Output += Input * Filter)
    // float* naive_input = new float[inputSize];
    // float* naive_output = new float[outputSize];
    // float* naive_filter = new float[filterSize];
    // float* naive_bias = new float[nOfm];
    // fill_random_array(naive_input, inputSize);
    // fill_random_array(naive_filter, filterSize);

    // // Backward Pass (Input += Output * Filter)
    // if (type == 'A' || type == 'B' || type == 'U') {
    //     float* naive_input_save = new float[inputSize];
    //     float* naive_output_bp = naive_output;
    //     for (int i = 0; i < inputSize; i++) {
    //         naive_input_save[i] = naive_input[i];
    //     }
    // }

    // // Update Weight (Filter += Input * Output)
    // if (type == 'A' || type == 'U') {
    //     float* naive_filter_wu = new float[filterSize];
    //     float* naive_output_wu = naive_output;
    // }

    // // Miscellaneous
    // /*
    // float* naive_output_save = new float[outputSize]
    // float* naive_filter_save = new float[filterSize];
    // float* naive_dbias = new float[nOfm]
    // for (size_t i = 0; i < outputSize; i++) {
    //     naive_output_save[i] = naive_output[i];
    // }
    // for (size_t i = 0; i < filterSize; i++) {
    //     naive_filter_save[i] = naive_filter[i];
    // }
    // */

    // /* CONV MEMORY */
    // // Forward Pass
    // float* conv_input = new float[inputSize];
    // float* conv_filter = new float[filterSize];
    // float* conv_output = new float[outputSize];
    // float* conv_bias = new float[nOfm];
    // for (size_t i = 0; i < inputSize; i++) {
    //     conv_input[i] = naive_input[i];
    // }
    // for (size_t i = 0; i < filterSize; i++) {
    //     conv_filter[i] = naive_filter[i];
    // }

    // // Backward Pass
    // if (type == 'A' || type == 'B' || type == 'U') {
    //     float* conv_output_bp = conv_output;
    //     float* conv_input_save = new float[inputSize];
    //     for (size_t i = 0; i < inputSize; i++) {
    //         conv_input_save[i] = naive_input[i];
    //     }
    // }

    // // Update Weight
    // if (type == 'A' || type == 'U') {
    //     float* conv_filter_wu = new float[filterSize];
    //     float* conv_output_wu = conv_output;
    // }
    
    /* Allocate memory for naive arrays */
    std::vector<float> naive_input(inputSize);
    std::vector<float> naive_input_save(inputSize);

    std::vector<float> naive_output(outputSize, 0.0f);
    std::vector<float> naive_output_bp = naive_output;
    std::vector<float> naive_output_wu = naive_output;

    std::vector<float> naive_filter(filterSize);
    std::vector<float> naive_filter_wu(filterSize);

    std::vector<float> naive_bias(nOfm);

    // Fill arrays with random values
    fill_random_array(naive_input, inputSize);
    fill_random_array(naive_filter, filterSize);

    // Copy values
    std::copy(naive_input.begin(), naive_input.end(), naive_input_save.begin());

    // Allocate memory for real convolutional arrays
    std::vector<float> conv_input(inputSize);
    std::vector<float> conv_input_save(inputSize);

    std::vector<float> conv_output(outputSize, 0.0f);
    std::vector<float> conv_output_bp = conv_output;
    std::vector<float> conv_output_wu = conv_output;

    std::vector<float> conv_filter(filterSize);
    std::vector<float> conv_filter_wu(filterSize);

    std::vector<float> conv_bias(nOfm);

    // Copy values
    std::copy(naive_input.begin(), naive_input.end(), conv_input.begin());
    std::copy(naive_input.begin(), naive_input.end(), conv_input_save.begin());
    std::copy(naive_filter.begin(), naive_filter.end(), conv_filter.begin());
 
    bool debug = false;

    if (debug) {
        cout << " DEBUGGING MODE " << endl;
        cout << "NAIVE vs. CONV INPUT\n" ;
        int error_count = 0;
        for (int i = 0; i < inputSize; i++) {
            if (naive_input[i] != conv_input[i])
                error_count++;
        }
        cout << "Error Count : " << error_count << endl;
        cout << "NAIVE vs. CONV FILTER\n" ;
        error_count = 0;
        for (int i = 0; i < filterSize; i++) {
            if (naive_filter_wu[i] != conv_filter_wu[i])
                error_count++;
        }
        cout << "Error Count : " << error_count << endl;
    }

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
    if (type == 'A' || type == 'F') { 
        start = high_resolution_clock::now();
        naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
        double l_total = (double)duration_sec.count();
        

        double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

        printf("Total Time = %.5g\n", (double)l_total);
        printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
        printf("fp time = %.5g\n", ((double)(l_total/iters)));
        printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);
        
    }
    if (type == 'A' || type == 'B') {
        naive_conv_bp(&naive_param, naive_input, naive_output_bp, naive_filter, naive_input_save);
    }
    if (type == 'A' || type == 'U') {
        naive_conv_uw(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);
    }

    printf("##########################################\n");
    printf("#           Performance Analysis         #\n");
    printf("##########################################\n");

    if (type == 'A' || type == 'F') {
        cout << "##########################################\n";
        cout << "               FORWARD PASS               \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        // naive_conv_fp(&conv_param, conv_input, conv_output, conv_filter, conv_bias);
        naive_conv_fp(&naive_param, conv_input, conv_output, conv_filter, conv_bias);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
        double l_total = (double)duration_sec.count() * 1e-3;
        

        double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

        printf("Total Time = %.5g\n", l_total);
        printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
        printf("fp time = %.5g\n", ((double)(l_total/iters)));
        printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

        // printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        //     ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        //     norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }
    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
        cout << "##########################################\n";
        cout << "               BACKWARD PASS              \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        reg_block_conv_bp(&conv_param, conv_input, conv_output_bp, conv_filter, conv_input_save);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
        double l_total = (double)duration_sec.count() * 1e-3;
        

        double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

        printf("Total Time = %.5g\n", (double)l_total);
        printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
        printf("fp time = %.5g\n", ((double)(l_total/iters)));
        printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);
    }
    if (type == 'A' || type == 'U') {
        cout << "##########################################\n";
        cout << "               UPDATE WEIGHT              \n";
        cout << "##########################################\n";

        start = high_resolution_clock::now();
        reg_block_conv_uw(&conv_param, conv_input_save, conv_output_wu, conv_filter_wu);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        //cout << "Total time consumed: " << duration_sec.count() << "ms\n";
        double l_total = (double)duration_sec.count() * 1e-3;
        

        double flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

        printf("Total Time = %.5g\n", (double)l_total);
        printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
        printf("fp time = %.5g\n", ((double)(l_total/iters)));
        printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);
    }

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

    if (debug) {
        cout << "DEBUGGING MODE" << endl;
        cout << "NAIVE vs. CONV OUTPUT" << endl;
        for (int i = 0; i < outputSize; i++) {
            cout << naive_output[i];
        }
        cout << endl;
        for (int i = 0; i < outputSize; i++) {
            cout << conv_output[i];
        }
        cout << endl;
    }
    printf("##########################################\n");
    printf("#           Cleaning Up data...          #\n");
    printf("##########################################\n");
    
    // //free allocated memory
    // /* NAIVE MEMORY*/
    // delete[] naive_input;
    // delete[] naive_output;
    // delete[] naive_filter;
    // delete[] naive_bias;
    // if (type == 'A' || type == 'B' || type = 'U') {
    //     delete[] naive_input_save;
    // }
    // if (type == 'A' || type == 'U') {
    //     delete[] naive_filter_wu;
    // }
    // //Miscellaneous
    // /*
    // delete[] naive_output_save;
    // delete[] naive_filter_save;
    // delete[] naive_dbias;
    // */
    
    // /* CONV MEMORY */
    // delete[] conv_input;
    // delete[] conv_output;
    // delete[] conv_filter;
    // delete[] conv_bias;
    // if (type == 'A' || type == 'B' || type = 'U') {
    //     delete[] conv_input_save;
    // }
    // if (type == 'A' || type == 'U') {
    //     delete[] conv_filter_wu;
    // }
    // //Miscellaneous
    // /*
    //     delete[] conv_output_save;
    //     delete[] conv_filter_save;
    //     delete[] conv_dbias;
    // */


    printf("##########################################\n");
    printf("#                Complete.               #\n");
    printf("##########################################\n");
    return 0;
}
 