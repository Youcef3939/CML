#include "sgd.h"
#include <stdlib.h>
#include "../tensor/tensor.h"
void sgd_step(Tensor* param, float lr) {
    if (!param || !param->grad) return;
    for (int i = 0; i < param->size; i++) {
        param->data[i] -= lr * param->grad[i];
    }
}
void sgd_step_params(Tensor** params, int n_params, float lr) {
    for (int i = 0; i < n_params; i++) {
        sgd_step(params[i], lr);
    }
}
void sgd_zero_grad(Tensor** params, int n_params) {
    for (int i = 0; i < n_params; i++) {
        if (!params[i] || !params[i]->grad) continue;

        for (int k = 0; k < params[i]->size; k++) {
            params[i]->grad[k] = 0.0f;
        }
    }
}
