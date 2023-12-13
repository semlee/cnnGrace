//regblock_ms2.h
#ifndef REGBLOCK_MS2_H
#define REGBLOCK_MS2_H

void reg_block_conv_fp(conv_t* param, const float* input, float* output, const float* filter, const float* bias);

void reg_block_conv_bp(conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save);

void reg_block_conv_uw(conv_t* param, const float* input, const float* output, float* filter);



#endif