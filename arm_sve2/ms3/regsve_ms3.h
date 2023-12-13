//regsve_ms3.h
#ifndef REGSVE_MS3_H
#define REGSVE_MS3_H

void arm_sve_conv_fp(conv_t* param, const float* input, float* output, const float* filter, const float* bias);

void arm_sve_conv_bp(conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save);

void arm_sve_conv_uw(conv_t* param, const float* input, const float* output, float* filter);



#endif