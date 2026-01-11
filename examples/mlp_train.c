#include <stdio.h>
#include <stdlib.h>
#include "tensor/tensor.h"
#include "data/csv.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/loss.h"
#include "optim/sgd.h"

int main() {
    Tensor* X = tensor_from_csv("data/train_X.csv"); 
    Tensor* y = tensor_from_csv("data/train_y.csv"); 

    printf("X shape: %d x %d\n", X->shape[0], X->shape[1]);
    printf("y shape: %d\n", y->shape[0]);

    Linear* fc1 = linear_create(X->shape[1], 4);
    Linear* fc2 = linear_create(4, 4);
    Linear* fc3 = linear_create(4, 2); 

    int epochs = 1000;
    float lr = 0.1f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* out1 = linear_forward(fc1, X);
        Tensor* act1 = relu(out1);

        Tensor* out2 = linear_forward(fc2, act1);
        Tensor* act2 = relu(out2);

        Tensor* logits = linear_forward(fc3, act2);

        Tensor* loss = cross_entropy_loss(logits, y);

        sgd_zero_grad((Tensor*[]){fc1->weight, fc1->bias,
                                  fc2->weight, fc2->bias,
                                  fc3->weight, fc3->bias}, 6);

        tensor_backward(loss);

        sgd_step_params((Tensor*[]){fc1->weight, fc1->bias,
                                    fc2->weight, fc2->bias,
                                    fc3->weight, fc3->bias}, 6, lr);

        if (epoch % 100 == 0) {
            printf("Epoch %d | Loss = %.6f\n", epoch, loss->data[0]);
            fflush(stdout);
        }

        tensor_release(out1);
        tensor_release(act1);
        tensor_release(out2);
        tensor_release(act2);
        tensor_release(logits);
        tensor_release(loss);
    }

    tensor_release(X);
    tensor_release(y);
    linear_free(fc1);
    linear_free(fc2);
    linear_free(fc3);

    return 0;
}
