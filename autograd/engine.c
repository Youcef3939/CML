#include "../tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    Tensor* tensor;          
    struct Node** children;  
    int n_children;
    int capacity;
} Node;

typedef struct {
    Node** nodes; 
    int count;
    int capacity;
} Graph;

static Graph engine = {0};

static Node* node_create(Tensor* t) {
    Node* n = (Node*)malloc(sizeof(Node));
    n->tensor = t;
    n->children = NULL;
    n->n_children = 0;
    n->capacity = 0;
    return n;
}

static void node_add_child(Node* parent, Tensor* child) {
    if (!parent || !child) return;

    if (parent->n_children == parent->capacity) {
        parent->capacity = parent->capacity ? parent->capacity*2 : 4;
        parent->children = (Node**)realloc(parent->children, sizeof(Node*) * parent->capacity);
    }

    Node* child_node = node_create(child);
    parent->children[parent->n_children++] = child_node;
}

static Node* engine_find(Tensor* t) {
    for (int i = 0; i < engine.count; i++) {
        if (engine.nodes[i]->tensor == t) return engine.nodes[i];
    }
    return NULL;
}

void engine_register(Tensor* t) {
    if (!t || !t->requires_grad) return;

    if (engine_find(t)) return; 

    if (engine.count == engine.capacity) {
        engine.capacity = engine.capacity ? engine.capacity*2 : 16;
        engine.nodes = (Node**)realloc(engine.nodes, sizeof(Node*) * engine.capacity);
    }

    Node* n = node_create(t);
    engine.nodes[engine.count++] = n;

    for (int i = 0; i < t->n_parents; i++)
        node_add_child(engine_find(t->parents[i]), t);
}

void engine_clear() {
    for (int i = 0; i < engine.count; i++) {
        Node* n = engine.nodes[i];
        for (int j = 0; j < n->n_children; j++)
            free(n->children[j]);
        free(n->children);
        free(n);
    }
    free(engine.nodes);
    engine.nodes = NULL;
    engine.count = 0;
    engine.capacity = 0;
}

typedef struct {
    Node** stack;
    int count;
    int capacity;
} NodeStack;

static void stack_push(NodeStack* s, Node* n) {
    if (s->count == s->capacity) {
        s->capacity = s->capacity ? s->capacity*2 : 16;
        s->stack = (Node**)realloc(s->stack, sizeof(Node*) * s->capacity);
    }
    s->stack[s->count++] = n;
}

static void topo_visit(Node* n, int* visited) {
    if (!n || visited[(size_t)n]) return;
    visited[(size_t)n] = 1;

    for (int i = 0; i < n->n_children; i++)
        topo_visit(n->children[i], visited);
}

void engine_backward(Tensor* loss) {
    if (!loss) return;

    engine_register(loss);

    tensor_backward(loss);

    engine_clear();
}

void engine_print() {
    printf("autograd engine\n");
    for (int i = 0; i < engine.count; i++) {
        Node* n = engine.nodes[i];
        printf("Tensor %p shape=[", (void*)n->tensor);
        for (int j = 0; j < n->tensor->ndim; j++) {
            printf("%d", n->tensor->shape[j]);
            if (j < n->tensor->ndim-1) printf(", ");
        }
        printf("], requires_grad=%d, children=%d\n", n->tensor->requires_grad, n->n_children);
    }
}