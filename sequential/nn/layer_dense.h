#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "tensor.h"

typedef struct {
    Tensor weights;
    Tensor bias;
    Tensor grad_w;
    Tensor grad_b;
    Tensor input_cache;
} DenseLayer;

void dense_init(DenseLayer *l, int in, int out);
void dense_forward(DenseLayer *l, Tensor *input, Tensor *output);
void dense_backward(DenseLayer *l, Tensor *grad_out, Tensor *grad_in);
void dense_free(DenseLayer *l);

#endif
