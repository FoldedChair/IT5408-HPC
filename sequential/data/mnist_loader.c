#include "mnist_loader.h"
#include "../utils/io.h"
#include <stdlib.h>

void load_fashion_mnist(Dataset *train, Dataset *test)
{
    train->images = read_idx_images(
        "data/train-images-idx3-ubyte",
        &train->num_samples
    );
    train->labels = read_idx_labels(
        "data/train-labels-idx1-ubyte",
        &train->num_samples
    );

    test->images = read_idx_images(
        "data/t10k-images-idx3-ubyte",
        &test->num_samples
    );
    test->labels = read_idx_labels(
        "data/t10k-labels-idx1-ubyte",
        &test->num_samples
    );
}

void free_dataset(Dataset *d)
{
    free(d->images);
    free(d->labels);
}
