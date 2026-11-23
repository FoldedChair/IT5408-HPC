#include "activations.h"
#include <math.h>

void relu_forward(Tensor *x, Tensor *out)
{
    int n = x->rows * x->cols;
    for (int i = 0; i < n; i++) {
        out->data[i] = (x->data[i] > 0.0f) ? x->data[i] : 0.0f;
    }
}

void relu_backward(Tensor *x, Tensor *grad_out, Tensor *grad_in)
{
    int n = x->rows * x->cols;
    for (int i = 0; i < n; i++) {
        grad_in->data[i] = (x->data[i] > 0.0f) ? grad_out->data[i] : 0.0f;
    }
}

void softmax(Tensor *x, Tensor *out)
{
    int n = x->rows * x->cols;

    float max = x->data[0];
    for (int i = 1; i < n; i++) {
        if (x->data[i] > max) max = x->data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out->data[i] = expf(x->data[i] - max);
        sum += out->data[i];
    }

    for (int i = 0; i < n; i++) {
        out->data[i] /= sum;
    }
}
