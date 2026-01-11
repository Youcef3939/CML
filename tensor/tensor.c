#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


static int compute_size(int ndim, const int* shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

static void compute_strides(int ndim, const int* shape, int* strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static float rand_uniform() {
    return (float)rand() / (float)RAND_MAX;
}

static float rand_normal() {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}


Tensor* tensor_create(int ndim, const int* shape, int requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = (int*)malloc(sizeof(int) * ndim);
    t->strides = (int*)malloc(sizeof(int) * ndim);

    memcpy(t->shape, shape, sizeof(int) * ndim);
    compute_strides(ndim, shape, t->strides);

    t->size = compute_size(ndim, shape);

    t->data = (float*)malloc(sizeof(float) * t->size);
    t->grad = requires_grad ? (float*)calloc(t->size, sizeof(float)) : NULL;

    t->parents = NULL;
    t->n_parents = 0;
    t->backward = NULL;

    t->requires_grad = requires_grad;
    t->is_view = 0;
    t->refcount = 1;

    return t;
}

Tensor* tensor_zeros(int ndim, const int* shape, int requires_grad) {
    Tensor* t = tensor_create(ndim, shape, requires_grad);
    if (!t) return NULL;

    memset(t->data, 0, sizeof(float) * t->size);
    return t;
}

Tensor* tensor_randn(int ndim, const int* shape, int requires_grad) {
    Tensor* t = tensor_create(ndim, shape, requires_grad);
    if (!t) return NULL;

    for (int i = 0; i < t->size; i++) {
        t->data[i] = rand_normal();
    }
    return t;
}


void tensor_retain(Tensor* t) {
    if (t) {
        t->refcount++;
    }
}

void tensor_release(Tensor* t) {
    if (!t) return;

    t->refcount--;
    if (t->refcount > 0) return;

    if (t->parents) {
        for (int i = 0; i < t->n_parents; i++) {
            tensor_release(t->parents[i]);
        }
        free(t->parents);
    }

    if (!t->is_view && t->data) {
        free(t->data);
    }

    if (t->grad) {
        free(t->grad);
    }

    free(t->shape);
    free(t->strides);
    free(t);
}


void tensor_zero_grad(Tensor* t) {
    if (!t || !t->grad) return;
    memset(t->grad, 0, sizeof(float) * t->size);
}

void tensor_print(const Tensor* t) {
    if (!t) return;

    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], requires_grad=%d)\n", t->requires_grad);

    for (int i = 0; i < t->size; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}