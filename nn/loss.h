#ifndef LOSS_H
#define LOSS_H
#include "../tensor/tensor.h"
Tensor* mse_loss(Tensor* predictions, Tensor* targets);
Tensor* cross_entropy_loss(Tensor* logits, Tensor* targets);
#endif