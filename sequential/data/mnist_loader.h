#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>

typedef struct {
    float *images;     /* [num_samples][784] */
    uint8_t *labels;   /* [num_samples] */
    int num_samples;
} Dataset;

void load_fashion_mnist(Dataset *train, Dataset *test);
void free_dataset(Dataset *d);

#endif
