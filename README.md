# CML
## machine learning in C because why not?

modern machine learning is conveniant but deeply opaque

you write 
```
loss.backward()
```

and trust that:
- gradients are correct

- memory is handled safely

- tensors are laid out efficently

- performance is "someone else's problem"

under the hood, that single line triggers:
- dynamic graph construction

- reverse mode automatic differentiation

- memory allocation and reuse

- kernel dispatch

- numerical edge cases you never see

most of the time it's great, until you actually want to **understand what's happening**

at some point abstractions stop helping and start hiding things
you read papers
you debug exploding gradients
you care about memory
you wonder why a model behaves the way it does

and the answer is usually "it's implimented in c/c++ somewhere inside the framework"

you open the source code and immmediately drown in massive codebases, GPU kernels and years of legacy abstractions!

understanding becomes harder than using


ML didn't start in python and didn't start in pytorch and certainly didn't start with one line APIs
at it's core it is just:
- linear algebra

- chain rule

- memory

so the questions came flowing:
- what if we removed the abstractions?

- what if we removed python?

- what if we build the whole thing in C?

not to be fast
not to be practical 
but to be honest

this decision became **CML**


## why does this exist?

most modern ML stacks look something like this:
```
C → CUDA → C++ → python → pytorch → your code
```
and then we call it a "high level AI"

so i asked
> if ML is already built on C anyway, why not cut the middle man?

**CML** became the answer

it isn't practical not fast and certainly not recommanded
it is however explicit, transparent and honest

## what is CML?

CML is a from-scratch ML framework written in pure C

no:
- pytorch
- tensorflow
- BLAS / LAPACK
- CUDA
- python
- external ML libraries

just:
- manual memory managment
- explicit linear algebra
- reverse mode automatic differentiation
- dynamic computation graphs
- consequences

## what CML actually impliments?

CML is intentionally small but **not trivial**

**CORE TENSOR SYSTEM:**
- N-dimensional tensors (float32)

- explicit shape and stride tracking

- contiguous memory layout

- tensor views (reshape / slice without copy)

- gradient buffers matching tensor shape

- reference-counted memory management

**AUTOMATIC DIFFERENTIATION:**
- reverse-mode autodiff (backpropagation)

- dynamic computation graphs

- each tensor tracks:

   - parent tensors

   - backward function pointer

- topological graph traversal during backward pass

- explicit gradient accumulation

no symbolic math, no magic
just graph construction and traversal

**MATHEMATICAL OPERATIONS:**
- elementwise ops: add, sub, mul

- matrix multiplication

- reduction ops (sum)

- broadcasting (limited, explicit)

- activation functions (Relu, sigmoid, tanh)

each operation:

- allocates a new tensor

- records its parents

- defines its own backward function

**NEURAL NETWORK LAYERS:**
- linear (dense) layers

- explicit parameter tensors (weights, bias)

- modular forward / backward logic

- gradient accumulation into parameters

and yes, this supports multi-layer perceptrons

**TRAINING UTILITIES:**
- loss functions:
    MSE
    BCE

- stochastic gradient descent

- mini batch training loop

- CSV dataset loader

yes, the loss actually decreases
no, it's nowhere fast

## what CML **deliberatly DOES NOT** do?

this is **NOT A PRODUCTION FRAMEWORK**, it's intentional

- no GPU support

- no CNNs/RNNs/transformers

- no BLAS/LAPACK

- no multithreading

- no checkpoints

- no python bindings

- no performance optimizations

if you want speed use **pytorch**

if you want comfort use **python**

and if you want to know what's really happening **keep reading**

## example usage
```
Tensor* x = tensor_randn(64, 10);
Tensor* y = model_forward(model, x);
Tensor* loss = mse(y, target);

tensor_backward(loss);
optimizer_step(opt);
```
it builds a computation graph
it backpropagates gradients

## want to give it a run?
```
gcc -o mlp_train examples/mlp_train.c tensor/tensor.c tensor/backward.c tensor/ops.c data/csv.c nn/linear.c nn/activations.c nn/loss.c optim/sgd.c autograd/engine.c -I. -Itensor -Idata -Inn -Ioptim -O2 -lm
```
then
```
./mlp_train.exe
```

## results

the network was trained on the XOR dataset (4 samples, 2 input features, 1 output)

below is a demonstration of the training progress over epochs with the cross-entropy loss decreasing as the network “learns”:

![alt text](<result.png>)

## MY design philosophy

explicit       > clever
readable       > optimized
correct        > fast
understandable > conveniance

every abstraction exists because it is necessary not because it is fashionable

## why C?
1. ML math is already implimented in C/C++

2. memory layout matters

3. abstractions leak

4. buidling things from scratch removes excuses

---

## diagram

```mermaid
flowchart TD

subgraph group_data["Data I/O"]
  node_csv["CSV Loader<br/>input parser<br/>[csv.c]"]
  node_train_x["Train X<br/>sample data<br/>[train_X.csv]"]
  node_train_y["Train Y<br/>sample data<br/>[train_y.csv]"]
end

subgraph group_core["Core Runtime"]
  node_tensor["Tensor<br/>core data structure<br/>[tensor.c]"]
  node_ops["Ops<br/>math kernels<br/>[ops.c]"]
  node_backward["Backward<br/>grad propagation<br/>[backward.c]"]
  node_engine["Autograd<br/>engine<br/>[engine.c]"]
end

subgraph group_nn["Neural Net"]
  node_linear["Linear<br/>dense layer<br/>[linear.c]"]
  node_activations["Activations<br/>nonlinear ops<br/>[activations.c]"]
end

subgraph group_train["Training"]
  node_loss["Losses<br/>objectives<br/>[loss.c]"]
  node_sgd["SGD<br/>optimizer<br/>[sgd.c]"]
end

subgraph group_app["Example"]
  node_mlp_train["MLP Train<br/>example app<br/>[mlp_train.c]"]
end

node_train_x -->|"read"| node_csv
node_train_y -->|"read"| node_csv
node_csv -->|"wrap"| node_tensor
node_tensor -->|"compute"| node_ops
node_ops -->|"feed"| node_linear
node_linear -->|"transform"| node_activations
node_activations -->|"score"| node_loss
node_loss -->|"backprop"| node_engine
node_engine -->|"dispatch"| node_backward
node_backward -->|"accumulate"| node_tensor
node_tensor -->|"params"| node_sgd
node_sgd -->|"update"| linear
node_mlp_train -->|"load"| node_csv
node_mlp_train -->|"build"| node_tensor
node_mlp_train -->|"model"| node_linear
node_mlp_train -->|"loss"| node_loss
node_mlp_train -->|"train"| node_engine
node_mlp_train -->|"step"| node_sgd

click node_tensor "https://github.com/youcef3939/cml/blob/main/tensor/tensor.c"
click node_ops "https://github.com/youcef3939/cml/blob/main/tensor/ops.c"
click node_backward "https://github.com/youcef3939/cml/blob/main/tensor/backward.c"
click node_engine "https://github.com/youcef3939/cml/blob/main/autograd/engine.c"
click node_csv "https://github.com/youcef3939/cml/blob/main/data/csv.c"
click node_linear "https://github.com/youcef3939/cml/blob/main/nn/linear.c"
click node_activations "https://github.com/youcef3939/cml/blob/main/nn/activations.c"
click node_loss "https://github.com/youcef3939/cml/blob/main/nn/loss.c"
click node_sgd "https://github.com/youcef3939/cml/blob/main/optim/sgd.c"
click node_mlp_train "https://github.com/youcef3939/cml/blob/main/examples/mlp_train.c"
click node_train_x "https://github.com/youcef3939/cml/blob/main/data/train_X.csv"
click node_train_y "https://github.com/youcef3939/cml/blob/main/data/train_y.csv"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_csv,node_train_x,node_train_y toneBlue
class node_tensor,node_ops,node_backward,node_engine toneAmber
class node_linear,node_activations toneMint
class node_loss,node_sgd toneRose
class node_mlp_train toneIndigo
```
---

> "you don't really understand something until you build it in C"
