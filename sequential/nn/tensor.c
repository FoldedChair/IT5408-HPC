#include "tensor.h"
#include "../utils/memory.h"

Tensor tensor_create(int rows, int cols)
{
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.data = alloc_f32(rows * cols);
    return t;
}

void tensor_free(Tensor *t)
{
    safe_free(t->data);
}

void tensor_zero(Tensor *t)
{
    for (int i = 0; i < t->rows * t->cols; i++)
        t->data[i] = 0.0f;
}
