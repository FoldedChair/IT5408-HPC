#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

void relu_forward(Tensor *x, Tensor *out);
void relu_backward(Tensor *x, Tensor *grad_out, Tensor *grad_in);
void softmax(Tensor *x, Tensor *out);

#endif
