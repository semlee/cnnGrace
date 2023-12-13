// naive_ms1.h
#ifndef NAIVE_MS1_H
#define NAIVE_MS1_H

void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias);

void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save);

void naive_conv_uw(naive_conv_t* param, const float* input, const float* output, float* filter);

#endif