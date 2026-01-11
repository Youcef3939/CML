#ifndef CML_ACTIVATIONS_H
#define CML_ACTIVATIONS_H
#include "../tensor/tensor.h"
Tensor* relu(Tensor* x);
Tensor* sigmoid(Tensor* x);
Tensor* softmax(Tensor* x);
Tensor* relu_backward(Tensor* grad_output, Tensor* x);
Tensor* sigmoid_backward(Tensor* grad_output, Tensor* x);
#endif