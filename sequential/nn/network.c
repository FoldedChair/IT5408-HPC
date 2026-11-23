#include "network.h"
#include "activations.h"
#include "loss.h"
#include "optimizer_sgd.h"
#include "../config.h"
#include <stdio.h>
#include <time.h>

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

    clock_t total_start = clock();
    double total_time = 0.0;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float total_loss = 0.0f;

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
            sgd_update(&net->l3, LEARNING_RATE);
            sgd_update(&net->l2, LEARNING_RATE);
            sgd_update(&net->l1, LEARNING_RATE);
        }
        
        clock_t epoch_end = clock();
        double epoch_time =
            (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;

        total_time += epoch_time;

        printf("Epoch %d | Loss: %.4f | Time: %.2f s\n",
               epoch + 1,
               total_loss / d->num_samples,
               epoch_time);
    }

    clock_t total_end = clock();
    double measured_total =
        (double)(total_end - total_start) / CLOCKS_PER_SEC;

    printf("Total training time: %.2f s\n", measured_total);

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
