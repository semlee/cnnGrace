//regblock_ms2.h
#ifndef REGBLOCK_MS2_H
#define REGBLOCK_MS2_H

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

void arm_sve_conv_fp(conv_t* param, const float* input, float* output, const float* filter, const float* bias);

void arm_sve_conv_bp(conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save);

void arm_sve_conv_uw(conv_t* param, const float* input, const float* output, float* filter);



#endif