#define _POSIX_C_SOURCE 200809L
#include "network.h"
#include "activations.h"
#include "loss.h"
#include "optimizer_sgd.h"
#include "../config.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

static double wall_time(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

void network_init(Network *net)
{
    dense_init(&net->l1, INPUT_SIZE, 256);
    dense_init(&net->l2, 256, 128);
    dense_init(&net->l3, 128, NUM_CLASSES);
}

void network_train(Network *net, Dataset *d)
{
    Tensor x      = tensor_create(INPUT_SIZE, 1);

    Tensor z1     = tensor_create(256, 1);
    Tensor a1     = tensor_create(256, 1);

    Tensor z2     = tensor_create(128, 1);
    Tensor a2     = tensor_create(128, 1);

    Tensor z3     = tensor_create(NUM_CLASSES, 1);
    Tensor probs  = tensor_create(NUM_CLASSES, 1);
    Tensor grad_z3 = tensor_create(NUM_CLASSES, 1);
    Tensor grad_a2 = tensor_create(128, 1);
    Tensor grad_z2 = tensor_create(128, 1);
    Tensor grad_a1 = tensor_create(256, 1);
    Tensor grad_z1 = tensor_create(256, 1);

    double total_start = wall_time();

    const char *epoch_env = getenv("TRAIN_EPOCHS");
    int epochs = epoch_env && atoi(epoch_env) > 0 ? atoi(epoch_env) : EPOCHS;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_start = wall_time();
        float total_loss = 0.0f;
        tensor_zero(&net->l1.grad_w); tensor_zero(&net->l1.grad_b);
        tensor_zero(&net->l2.grad_w); tensor_zero(&net->l2.grad_b);
        tensor_zero(&net->l3.grad_w); tensor_zero(&net->l3.grad_b);

        for (int i = 0; i < d->num_samples; i++) {

            /* Load input */
            for (int j = 0; j < INPUT_SIZE; j++) {
                x.data[j] = d->images[i * INPUT_SIZE + j];
            }

            /* Forward */
            dense_forward(&net->l1, &x,  &z1);
            relu_forward(&z1, &a1);

            dense_forward(&net->l2, &a1, &z2);
            relu_forward(&z2, &a2);

            dense_forward(&net->l3, &a2, &z3);
            softmax(&z3, &probs);

            int label = d->labels[i];
            total_loss += cross_entropy_loss(&probs, label);

            /* Backward */
            cross_entropy_backward(&probs, label, &grad_z3);
            dense_backward(&net->l3, &grad_z3, &grad_a2);
            relu_backward(&z2, &grad_a2, &grad_z2);
            dense_backward(&net->l2, &grad_z2, &grad_a1);
            relu_backward(&z1, &grad_a1, &grad_z1);
            dense_backward(&net->l1, &grad_z1, &x);

            /* Update */
            if ((i + 1) % BATCH_SIZE == 0 || i + 1 == d->num_samples) {
                int count = (i % BATCH_SIZE) + 1;
                sgd_update(&net->l3, LEARNING_RATE / count);
                sgd_update(&net->l2, LEARNING_RATE / count);
                sgd_update(&net->l1, LEARNING_RATE / count);
                tensor_zero(&net->l1.grad_w); tensor_zero(&net->l1.grad_b);
                tensor_zero(&net->l2.grad_w); tensor_zero(&net->l2.grad_b);
                tensor_zero(&net->l3.grad_w); tensor_zero(&net->l3.grad_b);
            }
        }
        
        double epoch_time = wall_time() - epoch_start;

        printf("Epoch %d | Loss: %.6f | Time: %.2f s\n",
               epoch + 1,
               total_loss / d->num_samples,
               epoch_time);
    }

    double measured_total = wall_time() - total_start;

    printf("Total training time: %.2f s\n", measured_total);
    const char *dump_prefix = getenv("DUMP_PARAMETERS");
    if (dump_prefix) {
        DenseLayer *layers[] = {&net->l1, &net->l2, &net->l3};
        for (int k = 0; k < 3; k++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s%d.bin", dump_prefix, k + 1);
            FILE *file = fopen(path, "wb");
            if (!file) { perror(path); exit(EXIT_FAILURE); }
            size_t n = (size_t)layers[k]->weights.rows * layers[k]->weights.cols;
            size_t b = (size_t)layers[k]->bias.rows;
            if (fwrite(layers[k]->weights.data, sizeof(float), n, file) != n ||
                fwrite(layers[k]->bias.data, sizeof(float), b, file) != b || fclose(file)) {
                fprintf(stderr, "Cannot write %s\n", path); exit(EXIT_FAILURE);
            }
        }
    }
    if (getenv("CHECK_WEIGHTS")) {
        DenseLayer *layers[] = {&net->l1, &net->l2, &net->l3};
        for (int k = 0; k < 3; k++) {
            double sum = 0.0, squares = 0.0;
            int n = layers[k]->weights.rows * layers[k]->weights.cols;
            for (int j = 0; j < n; j++) {
                double v = layers[k]->weights.data[j];
                sum += v; squares += v * v;
            }
            printf("Layer %d weights: %.12g %.12g\n", k + 1, sum, squares);
        }
    }

    tensor_free(&x);
    tensor_free(&z1);
    tensor_free(&a1);
    tensor_free(&a2);
    tensor_free(&z2);
    tensor_free(&z3);
    tensor_free(&probs);    
    tensor_free(&grad_z3);
    tensor_free(&grad_a2);
    tensor_free(&grad_z2);
    tensor_free(&grad_a1);
    tensor_free(&grad_z1);
}

void network_evaluate(Network *net, Dataset *d)
{
    Tensor x  = tensor_create(INPUT_SIZE, 1);
    Tensor z1 = tensor_create(256, 1);
    Tensor a1 = tensor_create(256, 1);
    Tensor z2 = tensor_create(128, 1);
    Tensor a2 = tensor_create(128, 1);
    Tensor z3    = tensor_create(NUM_CLASSES, 1);
    Tensor probs = tensor_create(NUM_CLASSES, 1);

    int correct = 0;

    for (int i = 0; i < d->num_samples; i++) {

        for (int j = 0; j < INPUT_SIZE; j++)
            x.data[j] = d->images[i * INPUT_SIZE + j];

        /* Forward */
        dense_forward(&net->l1, &x,  &z1);
        relu_forward(&z1, &a1);

        dense_forward(&net->l2, &a1, &z2);
        relu_forward(&z2, &a2);

        dense_forward(&net->l3, &a2, &z3);
        softmax(&z3, &probs);

        int pred = 0;
        float max = probs.data[0];
        for (int k = 1; k < NUM_CLASSES; k++) {
            if (probs.data[k] > max) {
                max = probs.data[k];
                pred = k;
            }
        }

        if (pred == d->labels[i])
            correct++;
    }

    printf("Accuracy: %.2f%%\n",
           100.0f * correct / d->num_samples);

    tensor_free(&x);
    tensor_free(&z1);
    tensor_free(&a1);
    tensor_free(&z2);
    tensor_free(&a2);
    tensor_free(&z3);
    tensor_free(&probs);
}

void network_free(Network *net)
{
    dense_free(&net->l1);
    dense_free(&net->l2);
    dense_free(&net->l3);
}
