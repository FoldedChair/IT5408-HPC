#include "loss.h"
#include "../config.h"
#include <math.h>

float cross_entropy_loss(Tensor *probs, int label)
{
    float p = probs->data[label];
    if (p < EPSILON) p = EPSILON;
    return -logf(p);
}

void cross_entropy_backward(Tensor *probs, int label, Tensor *grad)
{
    int n = probs->rows * probs->cols;

    for (int i = 0; i < n; i++)
        grad->data[i] = probs->data[i];

    grad->data[label] -= 1.0f;
}
