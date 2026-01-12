#include "loss.h"
#include <stdlib.h>
#include <stdio.h>
#include "../tensor/tensor.h"

static Tensor* flatten_targets(Tensor* t) {

    if (t->ndim == 1) return t;

    if (t->ndim == 2 && t->shape[1] == 1) {
        int N = t->shape[0];
        Tensor* out = tensor_zeros(1, &N, 0);
        for (int i = 0; i < N; i++)
            out->data[i] = t->data[i];
        return out;
    }

    fprintf(stderr, "cross_entropy_loss: unsupported target shape\n");
    exit(1);
}
typedef struct {
    Tensor* predictions;
    Tensor* targets;
    int N;
} MSEContext;

void mse_backward(Tensor* self) {
    MSEContext* ctx = (MSEContext*)self->backward;
    Tensor* pred = ctx->predictions;
    Tensor* targ = ctx->targets;
    int N = ctx->N;

    Tensor* diff = tensor_sub(pred, targ);
    Tensor* grad = tensor_mul_scalar(diff, 2.0f / (float)N);

    if (pred->grad) {
        for (int i = 0; i < pred->size; i++)
            pred->grad[i] += grad->data[i];
    }

    tensor_release(diff);
    tensor_release(grad);
    free(ctx);
}

Tensor* mse_loss(Tensor* predictions, Tensor* targets) {
    int N = predictions->shape[0];

    Tensor* diff = tensor_sub(predictions, targets);
    Tensor* sq = tensor_mul(diff, diff);
    Tensor* sum = tensor_sum(sq);
    Tensor* loss = tensor_div_scalar(sum, (float)N);

    MSEContext* ctx = malloc(sizeof(MSEContext));
    ctx->predictions = predictions;
    ctx->targets = targets;
    ctx->N = N;

    loss->backward = (void*)ctx;

    tensor_release(diff);
    tensor_release(sq);
    tensor_release(sum);

    return loss;
}

typedef struct {
    Tensor* logits;
    Tensor* targets;  
    int N;
    int C;
} CEContext;

void ce_backward(Tensor* self) {
    CEContext* ctx = (CEContext*)self->backward;
    Tensor* logits = ctx->logits;
    Tensor* targets = ctx->targets;
    int N = ctx->N;
    int C = ctx->C;

    Tensor* probs = tensor_softmax(logits);

    for (int i = 0; i < N; i++) {
        int idx = (int)targets->data[i];
        probs->data[i * C + idx] -= 1.0f;
    }

    Tensor* grad = tensor_div_scalar(probs, (float)N);

    if (logits->grad) {
        for (int i = 0; i < logits->size; i++)
            logits->grad[i] += grad->data[i];
    }

    tensor_release(probs);
    tensor_release(grad);
    free(ctx);
}

Tensor* cross_entropy_loss(Tensor* logits, Tensor* targets) {
    int N = logits->shape[0];
    int C = logits->shape[1];

    Tensor* flat_targets = flatten_targets(targets);

    Tensor* max_logits = tensor_max_axis(logits, 1);

    Tensor* shifted = tensor_zeros(2, logits->shape, logits->requires_grad);
    for (int i = 0; i < logits->shape[0]; i++) {
        for (int j = 0; j < logits->shape[1]; j++) {
            shifted->data[i * logits->shape[1] + j] =
                logits->data[i * logits->shape[1] + j] - max_logits->data[i];
    }
}

    
    Tensor* exp_logits = tensor_exp(shifted);
    Tensor* sum_exp = tensor_sum_axis(exp_logits, 1);
    Tensor* log_sum = tensor_log(sum_exp);
    Tensor* target_logits = tensor_gather(logits, flat_targets);
    Tensor* diff = tensor_sub(log_sum, target_logits);
    Tensor* loss_sum = tensor_sum(diff);
    Tensor* loss = tensor_div_scalar(loss_sum, (float)N);

    CEContext* ctx = malloc(sizeof(CEContext));
    ctx->logits = logits;
    ctx->targets = flat_targets; 
    ctx->N = N;
    ctx->C = C;

    loss->backward = (void*)ctx;

    tensor_release(max_logits);
    tensor_release(shifted);
    tensor_release(exp_logits);
    tensor_release(sum_exp);
    tensor_release(log_sum);
    tensor_release(target_logits);
    tensor_release(diff);
    tensor_release(loss_sum);

    return loss;
}
