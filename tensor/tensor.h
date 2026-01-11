#ifndef CML_TENSOR_H
#define CML_TENSOR_H
#include <stddef.h>

typedef struct Tensor Tensor;
struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int* strides;
    int ndim;
    int size;
    Tensor** parents;
    int n_parents;
    void (*backward)(Tensor* self);
    int requires_grad;
    int is_view;
    int refcount;
};
Tensor* tensor_create(int ndim, const int* shape, int requires_grad);
Tensor* tensor_zeros(int ndim, const int* shape, int requires_grad);
Tensor* tensor_randn(int ndim, const int* shape, int requires_grad);
void tensor_retain(Tensor* t);
void tensor_release(Tensor* t);
void tensor_zero_grad(Tensor* t);
void tensor_print(const Tensor* t);
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_sum(Tensor* a);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul_scalar(Tensor* a, float scalar);
Tensor* tensor_div_scalar(Tensor* a, float scalar);
Tensor* tensor_softmax(Tensor* a);
Tensor* tensor_max_axis(Tensor* a, int axis);
Tensor* tensor_sub_broadcast(Tensor* a, Tensor* b);
Tensor* tensor_exp(Tensor* a);
Tensor* tensor_sum_axis(Tensor* a, int axis);
Tensor* tensor_log(Tensor* a);
Tensor* tensor_gather(Tensor* a, Tensor* indices);
Tensor* tensor_add_broadcast(Tensor* a, Tensor* b);
Tensor* tensor_reshape(Tensor* a, int* new_shape, int new_ndim);
void tensor_backward(Tensor* loss);
#endif 