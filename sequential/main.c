#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "data/mnist_loader.h"
#include "nn/network.h"

int main(void)
{
    Dataset train, test;

    load_fashion_mnist(&train, &test);


    Network net;
    network_init(&net);

    network_train(&net, &train);
    network_evaluate(&net, &test);

    network_free(&net);
    free_dataset(&train);
    free_dataset(&test);

    return 0;
}
