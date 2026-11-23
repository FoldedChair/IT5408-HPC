#include "layer_dense.h"
#include "../utils/math_utils.h"

void dense_init(DenseLayer *l, int in, int out)
{
    l->weights = tensor_create(out, in);
    l->bias    = tensor_create(out, 1);
    l->grad_w  = tensor_create(out, in);
    l->grad_b  = tensor_create(out, 1);
    l->input_cache = tensor_create(in, 1);

    for (int i = 0; i < out * in; i++)
        l->weights.data[i] = rand_uniform(-0.05f, 0.05f);

    tensor_zero(&l->bias);
}

void dense_forward(DenseLayer *l, Tensor *input, Tensor *output)
{
    int out = l->weights.rows;
    int in  = l->weights.cols;

    for (int i = 0; i < in; i++)
        l->input_cache.data[i] = input->data[i];

    for (int i = 0; i < out; i++) {
        float sum = l->bias.data[i];

        for (int j = 0; j < in; j++) {
            sum += l->weights.data[i * in + j] * input->data[j];
        }

        output->data[i] = sum;
    }
}

void dense_backward(DenseLayer *l, Tensor *grad_out, Tensor *grad_in)
{
    int out = l->weights.rows;
    int in  = l->weights.cols;

    /* dW = grad_out * input^T */
    for (int i = 0; i < out; i++) {
        for (int j = 0; j < in; j++) {
            l->grad_w.data[i * in + j] =
                grad_out->data[i] * l->input_cache.data[j];
        }
    }

    /* db */
    for (int i = 0; i < out; i++) {
        l->grad_b.data[i] = grad_out->data[i];
    }

    /* dInput = W^T * grad_out */
    for (int j = 0; j < in; j++) {
        float sum = 0.0f;
        for (int i = 0; i < out; i++) {
            sum += l->weights.data[i * in + j] * grad_out->data[i];
        }
        grad_in->data[j] = sum;
    }
}

void dense_free(DenseLayer *l)
{
    tensor_free(&l->weights);
    tensor_free(&l->bias);
    tensor_free(&l->grad_w);
    tensor_free(&l->grad_b);
    tensor_free(&l->input_cache);
}
