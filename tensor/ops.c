#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static int check_same_shape(Tensor* a, Tensor* b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}
static void add_parent(Tensor* t, Tensor* parent) {
    t->parents = (Tensor**)realloc(t->parents, sizeof(Tensor*) * (t->n_parents + 1));
    t->parents[t->n_parents] = parent;
    t->n_parents++;
    tensor_retain(parent);
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!check_same_shape(a, b)) { fprintf(stderr, "tensor_add shape mismatch\n"); return NULL; }
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] + b->data[i];
    if (out->requires_grad) { add_parent(out, a); add_parent(out, b); out->backward = NULL; }
    return out;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!check_same_shape(a, b)) { fprintf(stderr, "tensor_sub shape mismatch\n"); return NULL; }
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] - b->data[i];
    if (out->requires_grad) { add_parent(out, a); add_parent(out, b); out->backward = NULL; }
    return out;
}
Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!check_same_shape(a, b)) { fprintf(stderr, "tensor_mul shape mismatch\n"); return NULL; }
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] * b->data[i];
    if (out->requires_grad) { add_parent(out, a); add_parent(out, b); out->backward = NULL; }
    return out;
}

Tensor* tensor_mul_scalar(Tensor* a, float scalar) {
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] * scalar;
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_div_scalar(Tensor* a, float scalar) {
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] / scalar;
    if (out->requires_grad) add_parent(out, a);
    return out;
}
Tensor* tensor_sum(Tensor* a) {
    Tensor* out = tensor_zeros(0, NULL, a->requires_grad);
    float total = 0.0f;
    for (int i = 0; i < a->size; i++) total += a->data[i];
    out->data[0] = total;
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_sum_axis(Tensor* a, int axis) {
    if (a->ndim != 2) { fprintf(stderr, "tensor_sum_axis only supports 2D tensors\n"); return NULL; }
    if (axis < 0 || axis > 1) { fprintf(stderr, "tensor_sum_axis invalid axis\n"); return NULL; }
    int out_shape[1] = { axis == 0 ? a->shape[1] : a->shape[0] };
    Tensor* out = tensor_zeros(1, out_shape, a->requires_grad);

    if (axis == 0) {
        for (int j = 0; j < a->shape[1]; j++) {
            float s = 0.0f;
            for (int i = 0; i < a->shape[0]; i++) s += a->data[i * a->shape[1] + j];
            out->data[j] = s;
        }
    } else {
        for (int i = 0; i < a->shape[0]; i++) {
            float s = 0.0f;
            for (int j = 0; j < a->shape[1]; j++) s += a->data[i * a->shape[1] + j];
            out->data[i] = s;
        }
    }
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) { fprintf(stderr, "tensor_matmul dimension mismatch\n"); return NULL; }
    int m = a->shape[0], n = a->shape[1], p = b->shape[1];
    int out_shape[2] = { m, p };
    Tensor* out = tensor_zeros(2, out_shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) sum += a->data[i*n + k] * b->data[k*p + j];
            out->data[i*p + j] = sum;
        }
    if (out->requires_grad) { add_parent(out, a); add_parent(out, b); out->backward = NULL; }
    return out;
}
Tensor* tensor_exp(Tensor* a) {
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = expf(a->data[i]);
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* out = tensor_zeros(a->ndim, a->shape, a->requires_grad);
    for (int i = 0; i < a->size; i++) out->data[i] = logf(a->data[i]);
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_max_axis(Tensor* a, int axis) {
    if (a->ndim != 2) { fprintf(stderr, "tensor_max_axis only supports 2D tensors\n"); return NULL; }
    int out_shape[1] = { axis == 0 ? a->shape[1] : a->shape[0] };
    Tensor* out = tensor_zeros(1, out_shape, 0);

    if (axis == 0) {
        for (int j = 0; j < a->shape[1]; j++) {
            float maxv = a->data[j];
            for (int i = 1; i < a->shape[0]; i++) if (a->data[i*a->shape[1]+j] > maxv) maxv = a->data[i*a->shape[1]+j];
            out->data[j] = maxv;
        }
    } else {
        for (int i = 0; i < a->shape[0]; i++) {
            float maxv = a->data[i*a->shape[1]];
            for (int j = 1; j < a->shape[1]; j++) if (a->data[i*a->shape[1]+j] > maxv) maxv = a->data[i*a->shape[1]+j];
            out->data[i] = maxv;
        }
    }
    return out;
}

Tensor* tensor_sub_broadcast(Tensor* a, Tensor* b) {
    if (a->ndim != 2 || b->ndim != 1 || a->shape[1] != b->shape[0]) { fprintf(stderr,"tensor_sub_broadcast shape mismatch\n"); return NULL; }
    int out_shape[2] = {a->shape[0], a->shape[1]};
    Tensor* out = tensor_zeros(2, out_shape, a->requires_grad);
    for (int i = 0; i < a->shape[0]; i++)
        for (int j = 0; j < a->shape[1]; j++)
            out->data[i*a->shape[1]+j] = a->data[i*a->shape[1]+j] - b->data[j];
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_add_broadcast(Tensor* a, Tensor* b) {
    if (check_same_shape(a, b)) return tensor_add(a, b);

    if (a->ndim == 2 && b->ndim == 1 && a->shape[1] == b->shape[0]) {
        int out_shape[2] = { a->shape[0], a->shape[1] };
        Tensor* out = tensor_zeros(2, out_shape, a->requires_grad || b->requires_grad);
        for (int i = 0; i < a->shape[0]; i++)
            for (int j = 0; j < a->shape[1]; j++)
                out->data[i*a->shape[1]+j] = a->data[i*a->shape[1]+j] + b->data[j];
        if (out->requires_grad) { add_parent(out, a); add_parent(out, b); }
        return out;
    }

    fprintf(stderr,"tensor_add_broadcast shape mismatch\n");
    return NULL;
}

Tensor* tensor_reshape(Tensor* a, int* new_shape, int new_ndim) {
    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) new_size *= new_shape[i];
    if (new_size != a->size) { fprintf(stderr, "tensor_reshape size mismatch\n"); return NULL; }

    Tensor* out = (Tensor*)malloc(sizeof(Tensor));
    out->ndim = new_ndim;
    out->shape = (int*)malloc(sizeof(int)*new_ndim);
    memcpy(out->shape, new_shape, sizeof(int)*new_ndim);
    out->size = a->size;
    out->data = a->data;
    out->grad = a->grad;
    out->requires_grad = a->requires_grad;
    out->parents = NULL;
    out->n_parents = 0;
    out->backward = NULL;

    return out;
}

Tensor* tensor_softmax(Tensor* a) {
    if (a->ndim != 2) { fprintf(stderr, "tensor_softmax only supports 2D tensors\n"); return NULL; }
    int N = a->shape[0], C = a->shape[1];
    Tensor* out = tensor_zeros(2, a->shape, a->requires_grad);
    for (int i = 0; i < N; i++) {
        float maxv = a->data[i*C];
        for (int j = 1; j < C; j++) if (a->data[i*C+j] > maxv) maxv = a->data[i*C+j];
        float sum = 0.0f;
        for (int j = 0; j < C; j++) { out->data[i*C+j] = expf(a->data[i*C+j] - maxv); sum += out->data[i*C+j]; }
        for (int j = 0; j < C; j++) out->data[i*C+j] /= sum;
    }
    if (out->requires_grad) add_parent(out, a);
    return out;
}

Tensor* tensor_gather(Tensor* a, Tensor* indices) {
    if (a->ndim != 2 || indices->ndim != 1 || a->shape[0] != indices->shape[0]) { fprintf(stderr, "tensor_gather shape mismatch\n"); return NULL; }
    int N = a->shape[0];
    Tensor* out = tensor_zeros(1, &N, a->requires_grad);
    for (int i = 0; i < N; i++) { 
        int idx = (int)indices->data[i]; 
        out->data[i] = a->data[i*a->shape[1] + idx];
    }
    if (out->requires_grad) add_parent(out, a);
    return out;
}

float tensor_item(Tensor* t, int i) {
    if (t->size <= i) { fprintf(stderr, "tensor_item index out of bounds\n"); return 0; }
    return t->data[i];
}

void tensor_free(Tensor* t) {
    tensor_release(t);
}
