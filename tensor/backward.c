#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>


static void backward_add(Tensor* t) {
    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++)
            a->grad[i] += t->grad[i];
    }

    if (b->requires_grad) {
        for (int i = 0; i < b->size; i++)
            b->grad[i] += t->grad[i];
    }
}

static void backward_mul(Tensor* t) {
    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++)
            a->grad[i] += b->data[i] * t->grad[i];
    }

    if (b->requires_grad) {
        for (int i = 0; i < b->size; i++)
            b->grad[i] += a->data[i] * t->grad[i];
    }
}

static void backward_sum(Tensor* t) {
    Tensor* a = t->parents[0];
    if (!a->requires_grad) return;

    for (int i = 0; i < a->size; i++)
        a->grad[i] += t->grad[0]; 
}

static void backward_matmul(Tensor* t) {
    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];

    if (a->requires_grad) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < p; k++)
                    sum += t->grad[i*p + k] * b->data[j*p + k];
                a->grad[i*n + j] += sum;
            }
    }

    if (b->requires_grad) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++) {
                float sum = 0.0f;
                for (int k = 0; k < m; k++)
                    sum += a->data[k*n + i] * t->grad[k*p + j];
                b->grad[i*p + j] += sum;
            }
    }
}

typedef struct {
    Tensor** nodes;
    int count;
    int capacity;
} TensorStack;

static void stack_push(TensorStack* stack, Tensor* t) {
    if (stack->count == stack->capacity) {
        stack->capacity = stack->capacity ? stack->capacity*2 : 16;
        stack->nodes = (Tensor**)realloc(stack->nodes, sizeof(Tensor*) * stack->capacity);
    }
    stack->nodes[stack->count++] = t;
}

static void build_topo(Tensor* t, TensorStack* stack, int* visited) {
    if (!t || visited[(size_t)t]) return;
    visited[(size_t)t] = 1;
    for (int i = 0; i < t->n_parents; i++)
        build_topo(t->parents[i], stack, visited);
    stack_push(stack, t);
}

void tensor_backward(Tensor* loss) {
    if (!loss) return;

    if (!loss->grad) {
        loss->grad = (float*)calloc(loss->size, sizeof(float));
        for (int i = 0; i < loss->size; i++)
            loss->grad[i] = 1.0f; 
    }

    TensorStack stack = {0};
    int* visited = (int*)calloc(1024, sizeof(int));
    build_topo(loss, &stack, visited);
    free(visited);

    for (int i = stack.count - 1; i >= 0; i--) {
        Tensor* t = stack.nodes[i];
        if (!t->grad) t->grad = (float*)calloc(t->size, sizeof(float));

        if (t->backward) t->backward(t);
        else {
            if (t->n_parents == 2) backward_add(t);
            else if (t->n_parents == 1 && t->ndim == 0) backward_sum(t);
            else if (t->n_parents == 1) backward_matmul(t);
        }
    }

    free(stack.nodes);
}