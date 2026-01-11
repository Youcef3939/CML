#include <stdlib.h>
#include <stdio.h>
#include "../tensor/tensor.h"
#include "linear.h"

Linear* linear_create(int in_features, int out_features) {
    Linear* layer = (Linear*)malloc(sizeof(Linear));
    if (!layer) {
        fprintf(stderr, "failed to allocate Linear layer\n");
        exit(1);
    }

    layer->in_features = in_features;
    layer->out_features = out_features;

    int w_shape[2] = {in_features, out_features};
    int b_shape[1] = {out_features};

    layer->weight = tensor_randn(2, w_shape, 1);  
    layer->bias   = tensor_zeros(1, b_shape, 1); 

    return layer;
}

Tensor* linear_forward(Linear* layer, Tensor* x) {
    if (x->ndim != 2 || x->shape[1] != layer->in_features) {
        fprintf(stderr,
            "Linear forward shape mismatch: got [%d, %d], expected [*, %d]\n",
            x->shape[0], x->shape[1], layer->in_features
        );
        exit(1);
    }

    Tensor* out = tensor_matmul(x, layer->weight);  

    Tensor* bias_flat = layer->bias;
    if (layer->bias->ndim != 1) {
        int new_shape[1] = { layer->bias->shape[layer->bias->ndim - 1] };
        bias_flat = tensor_reshape(layer->bias, new_shape, 1);
    }

    Tensor* y = tensor_add_broadcast(out, bias_flat);  
    tensor_release(out);

    return y;
}

void linear_zero_grad(Linear* layer) {
    tensor_zero_grad(layer->weight);
    tensor_zero_grad(layer->bias);
}

void linear_free(Linear* layer) {
    if (!layer) return;
    tensor_release(layer->weight);
    tensor_release(layer->bias);
    free(layer);
}

void linear_print(Linear* layer) {
    printf("Linear(in=%d, out=%d)\n", layer->in_features, layer->out_features);
    printf("Weights:\n");
    tensor_print(layer->weight);
    printf("Bias:\n");
    tensor_print(layer->bias);
}