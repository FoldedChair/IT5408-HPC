#include "optimizer_sgd.h"

void sgd_update(DenseLayer *l, float lr)
{
    int n_w = l->weights.rows * l->weights.cols;

    for (int i = 0; i < n_w; i++)
        l->weights.data[i] -= lr * l->grad_w.data[i];

    for (int i = 0; i < l->bias.rows; i++)
        l->bias.data[i] -= lr * l->grad_b.data[i];
}
