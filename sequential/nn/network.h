#ifndef NETWORK_H
#define NETWORK_H

#include "layer_dense.h"
#include "../data/mnist_loader.h"

typedef struct {
    DenseLayer l1; /* 784 → 256 */
    DenseLayer l2; /* 256 → 128 */
    DenseLayer l3; /* 128 → 10  */
} Network;

void network_init(Network *net);
void network_train(Network *net, Dataset *d);
void network_evaluate(Network *net, Dataset *d);
void network_free(Network *net);

#endif
