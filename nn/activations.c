#include <stdlib.h>
#include <math.h>
#include "tensor.h"

static void relu_backward(Tensor* out) {
    Tensor* x = out->parents[0];
    if (!x->requires_grad) return;

    for (int i = 0; i < x->size; i++) {
        x->grad[i] += (x->data[i] > 0.0f) ? out->grad[i] : 0.0f;
    }
}

Tensor* relu(Tensor* x) {
    Tensor* out = tensor_create(x->ndim, x->shape, x->requires_grad);

    for (int i = 0; i < x->size; i++) {
        out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    }

    if (out->requires_grad) {
        out->parents = malloc(sizeof(Tensor*));
        out->parents[0] = x;
        out->n_parents = 1;
        out->backward = relu_backward;
        tensor_retain(x);
    }

    return out;
}

static void sigmoid_backward(Tensor* out) {
    Tensor* x = out->parents[0];
    if (!x->requires_grad) return;

    for (int i = 0; i < x->size; i++) {
        float y = out->data[i];
        x->grad[i] += out->grad[i] * y * (1.0f - y);
    }
}

Tensor* sigmoid(Tensor* x) {
    Tensor* out = tensor_create(x->ndim, x->shape, x->requires_grad);

    for (int i = 0; i < x->size; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-x->data[i]));
    }

    if (out->requires_grad) {
        out->parents = malloc(sizeof(Tensor*));
        out->parents[0] = x;
        out->n_parents = 1;
        out->backward = sigmoid_backward;
        tensor_retain(x);
    }

    return out;
}


static void tanh_backward(Tensor* out) {
    Tensor* x = out->parents[0];
    if (!x->requires_grad) return;

    for (int i = 0; i < x->size; i++) {
        float y = out->data[i];
        x->grad[i] += out->grad[i] * (1.0f - y * y);
    }
}

Tensor* tanh_tensor(Tensor* x) {
    Tensor* out = tensor_create(x->ndim, x->shape, x->requires_grad);

    for (int i = 0; i < x->size; i++) {
        out->data[i] = tanhf(x->data[i]);
    }

    if (out->requires_grad) {
        out->parents = malloc(sizeof(Tensor*));
        out->parents[0] = x;
        out->n_parents = 1;
        out->backward = tanh_backward;
        tensor_retain(x);
    }

    return out;
}