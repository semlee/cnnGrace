// naive_ms1.h
#ifndef NAIVE_MS1_H
#define NAIVE_MS1_H

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

void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias);

void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save) 

void naive_conv_uw(naive_conv_t* param, const float* input, const float* output, float* filter)



#endif