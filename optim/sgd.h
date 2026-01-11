#ifndef SGD_H
#define SGD_H
#include "../tensor/tensor.h"
void sgd_step(Tensor* param, float lr);
void sgd_step_params(Tensor** params, int n_params, float lr);
void sgd_zero_grad(Tensor** params, int n_params);
#endif