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

void reg_block_conv_fp(conv_t* param, const std::vector<float>& input, std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& bias);

void reg_block_conv_bp(conv_t* param, std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& filter, const std::vector<float>& naive_input_save);

void reg_block_conv_uw(conv_t* param, const std::vector<float>& input, const std::vector<float>& output, std::vector<float>& filter);



#endif