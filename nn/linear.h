#ifndef CML_LINEAR_H
#define CML_LINEAR_H
#include "../tensor/tensor.h"
typedef struct Linear Linear;
struct Linear {
    int in_features;
    int out_features;
    Tensor* weight;
    Tensor* bias;
};
Linear* linear_create(int input_dim, int output_dim);
Tensor* linear_forward(Linear* layer, Tensor* input);
void linear_zero_grad(Linear* layer);
void linear_free(Linear* layer);
void linear_print(Linear* layer);
#endif