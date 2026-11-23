#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

float cross_entropy_loss(Tensor *probs, int label);
void cross_entropy_backward(Tensor *probs, int label, Tensor *grad);

#endif
