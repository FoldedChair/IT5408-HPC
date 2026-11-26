#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "data/mnist_loader.h"
#include "nn/network.h"

int main(void)
{
    Dataset train, test;

    load_fashion_mnist(&train, &test);
    const char *limit = getenv("TRAIN_LIMIT");
    if (limit && atoi(limit) > 0 && atoi(limit) < train.num_samples)
        train.num_samples = atoi(limit);
    limit = getenv("TEST_LIMIT");
    if (limit && atoi(limit) > 0 && atoi(limit) < test.num_samples)
        test.num_samples = atoi(limit);


    struct timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);
    Network net;
    srand(42);
    network_init(&net);

    network_train(&net, &train);
    network_evaluate(&net, &test);
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    printf("Total runtime (initialization + training + evaluation): %.2f s\n",
           (double)(total_end.tv_sec - total_start.tv_sec) +
           (total_end.tv_nsec - total_start.tv_nsec) / 1e9);

    network_free(&net);
    free_dataset(&train);
    free_dataset(&test);

    return 0;
}
