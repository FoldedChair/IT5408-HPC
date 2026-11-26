/**
 * Pipeline Parallelism for Fully Connected Neural Network using MPI
 * 
 * Process 0: Input + Hidden Layer 1 (784 -> 256)
 * Process 1: Hidden Layer 2 (256 -> 128)
 * Process 2: Output Layer + Loss calculation (128 -> 10)
 * 
 * Dataset: MNIST (Fashion-MNIST compatible)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ========== Configuration ========== */
#define INPUT_SIZE     784
#define HIDDEN1_SIZE   256
#define HIDDEN2_SIZE   128
#define NUM_CLASSES    10

#define EPOCHS         10
#define LEARNING_RATE  0.01f
#define EPSILON        1e-7f

#define NUM_PROCESSES  3

/* MPI Tags */
#define TAG_FORWARD_A1     100
#define TAG_FORWARD_A2     101
#define TAG_FORWARD_LABEL  102
#define TAG_BACKWARD_GRAD1 200
#define TAG_BACKWARD_GRAD2 201
#define TAG_LOSS           300
#define TAG_SAMPLE_DONE    400

/* ========== Tensor Structure ========== */
typedef struct {
    float *data;
    int rows;
    int cols;
} Tensor;

Tensor tensor_create(int rows, int cols) {
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.data = (float *)calloc(rows * cols, sizeof(float));
    return t;
}

void tensor_free(Tensor *t) {
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void tensor_zero(Tensor *t) {
    memset(t->data, 0, t->rows * t->cols * sizeof(float));
}

/* ========== Dense Layer Structure ========== */
typedef struct {
    Tensor weights;
    Tensor bias;
    Tensor grad_w;
    Tensor grad_b;
    Tensor input_cache;
    int in_size;
    int out_size;
} DenseLayer;

float rand_uniform(float min, float max) {
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}

void dense_init(DenseLayer *l, int in, int out) {
    l->in_size = in;
    l->out_size = out;
    l->weights = tensor_create(out, in);
    l->bias = tensor_create(out, 1);
    l->grad_w = tensor_create(out, in);
    l->grad_b = tensor_create(out, 1);
    l->input_cache = tensor_create(in, 1);

    for (int i = 0; i < out * in; i++) {
        l->weights.data[i] = rand_uniform(-0.05f, 0.05f);
    }
    tensor_zero(&l->bias);
}

void dense_forward(DenseLayer *l, float *input, float *output) {
    int out = l->out_size;
    int in = l->in_size;

    /* Cache input for backward pass */
    memcpy(l->input_cache.data, input, in * sizeof(float));

    for (int i = 0; i < out; i++) {
        float sum = l->bias.data[i];
        for (int j = 0; j < in; j++) {
            sum += l->weights.data[i * in + j] * input[j];
        }
        output[i] = sum;
    }
}

void dense_backward(DenseLayer *l, float *grad_out, float *grad_in) {
    int out = l->out_size;
    int in = l->in_size;

    /* dW = grad_out * input^T */
    for (int i = 0; i < out; i++) {
        for (int j = 0; j < in; j++) {
            l->grad_w.data[i * in + j] += grad_out[i] * l->input_cache.data[j];
        }
    }

    /* db */
    for (int i = 0; i < out; i++) {
        l->grad_b.data[i] += grad_out[i];
    }

    /* dInput = W^T * grad_out */
    if (grad_in != NULL) {
        for (int j = 0; j < in; j++) {
            float sum = 0.0f;
            for (int i = 0; i < out; i++) {
                sum += l->weights.data[i * in + j] * grad_out[i];
            }
            grad_in[j] = sum;
        }
    }
}

void dense_free(DenseLayer *l) {
    tensor_free(&l->weights);
    tensor_free(&l->bias);
    tensor_free(&l->grad_w);
    tensor_free(&l->grad_b);
    tensor_free(&l->input_cache);
}

void sgd_update(DenseLayer *l, float lr) {
    int n_w = l->out_size * l->in_size;
    for (int i = 0; i < n_w; i++) {
        l->weights.data[i] -= lr * l->grad_w.data[i];
    }
    for (int i = 0; i < l->out_size; i++) {
        l->bias.data[i] -= lr * l->grad_b.data[i];
    }
}

/* ========== Activation Functions ========== */
void relu_forward(float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

void relu_backward(float *x, float *grad_out, float *grad_in, int n) {
    for (int i = 0; i < n; i++) {
        grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

void softmax(float *x, float *out, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

/* ========== Loss Functions ========== */
float cross_entropy_loss(float *probs, int label) {
    float p = probs[label];
    if (p < EPSILON) p = EPSILON;
    return -logf(p);
}

void cross_entropy_backward(float *probs, int label, float *grad, int n) {
    for (int i = 0; i < n; i++) {
        grad[i] = probs[i];
    }
    grad[label] -= 1.0f;
}

/* ========== Dataset Loading ========== */
typedef struct {
    float *images;
    uint8_t *labels;
    int num_samples;
} Dataset;

static uint32_t read_u32(FILE *f) {
    uint8_t b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

float *read_idx_images(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return NULL;
    }

    read_u32(f);               /* magic */
    *count = read_u32(f);
    int rows = read_u32(f);
    int cols = read_u32(f);

    int size = (*count) * rows * cols;
    float *data = (float *)malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        unsigned char pixel;
        fread(&pixel, 1, 1, f);
        data[i] = pixel / 255.0f;
    }

    fclose(f);
    return data;
}

uint8_t *read_idx_labels(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return NULL;
    }

    read_u32(f);   /* magic */
    *count = read_u32(f);

    uint8_t *labels = (uint8_t *)malloc(*count);
    fread(labels, 1, *count, f);

    fclose(f);
    return labels;
}

void load_mnist(Dataset *train, Dataset *test) {
    train->images = read_idx_images("../data/train-images-idx3-ubyte", &train->num_samples);
    train->labels = read_idx_labels("../data/train-labels-idx1-ubyte", &train->num_samples);
    test->images = read_idx_images("../data/t10k-images-idx3-ubyte", &test->num_samples);
    test->labels = read_idx_labels("../data/t10k-labels-idx1-ubyte", &test->num_samples);
}

void free_dataset(Dataset *d) {
    if (d->images) free(d->images);
    if (d->labels) free(d->labels);
}

/* ========== Process 0: Input + Hidden Layer 1 ========== */
void process_0_run(Dataset *train, Dataset *test, int num_samples) {
    DenseLayer l1;
    dense_init(&l1, INPUT_SIZE, HIDDEN1_SIZE);

    /* Buffers */
    float *x = (float *)malloc(INPUT_SIZE * sizeof(float));
    float *z1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *a1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *grad_a1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *grad_z1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));

    MPI_Status status;

    printf("[Process 0] Starting training with %d samples\n", num_samples);

    MPI_Barrier(MPI_COMM_WORLD);
    double train_start = MPI_Wtime();    

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_start = MPI_Wtime();

        for (int i = 0; i < num_samples; i++) {
            /* Load input */
            memcpy(x, &train->images[i * INPUT_SIZE], INPUT_SIZE * sizeof(float));
            int label = train->labels[i];

            /* Forward: Dense Layer 1 + ReLU */
            dense_forward(&l1, x, z1);
            relu_forward(z1, a1, HIDDEN1_SIZE);

            /* Send activation to Process 1 */
            MPI_Send(a1, HIDDEN1_SIZE, MPI_FLOAT, 1, TAG_FORWARD_A1, MPI_COMM_WORLD);
            /* Send label to Process 2 */
            MPI_Send(&label, 1, MPI_INT, 2, TAG_FORWARD_LABEL, MPI_COMM_WORLD);

            /* Receive gradient from Process 1 */
            MPI_Recv(grad_a1, HIDDEN1_SIZE, MPI_FLOAT, 1, TAG_BACKWARD_GRAD1, MPI_COMM_WORLD, &status);

            /* Backward */
            relu_backward(z1, grad_a1, grad_z1, HIDDEN1_SIZE);
            dense_backward(&l1, grad_z1, NULL);  /* No need to propagate further */

            /* Update weights */
            sgd_update(&l1, LEARNING_RATE);
        }

        double epoch_end = MPI_Wtime();

        /* Receive and print loss from Process 2 */
        float avg_loss;
        MPI_Recv(&avg_loss, 1, MPI_FLOAT, 2, TAG_LOSS, MPI_COMM_WORLD, &status);

        printf("Epoch %d | Loss: %.4f | Time: %.2f s\n", epoch + 1, avg_loss, epoch_end - epoch_start);
    }

    double train_elapsed = MPI_Wtime() - train_start;
    double max_train_elapsed;

    MPI_Reduce(&train_elapsed, &max_train_elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("\nMPI Training time: %.2f seconds\n", max_train_elapsed);

    /* ========== Evaluation ========== */
    printf("[Process 0] Starting evaluation with %d samples\n", test->num_samples);

    for (int i = 0; i < test->num_samples; i++) {
        memcpy(x, &test->images[i * INPUT_SIZE], INPUT_SIZE * sizeof(float));
        int label = test->labels[i];

        dense_forward(&l1, x, z1);
        relu_forward(z1, a1, HIDDEN1_SIZE);

        MPI_Send(a1, HIDDEN1_SIZE, MPI_FLOAT, 1, TAG_FORWARD_A1, MPI_COMM_WORLD);
        MPI_Send(&label, 1, MPI_INT, 2, TAG_FORWARD_LABEL, MPI_COMM_WORLD);
    }

    /* Receive accuracy from Process 2 */
    float accuracy;
    MPI_Recv(&accuracy, 1, MPI_FLOAT, 2, TAG_LOSS, MPI_COMM_WORLD, &status);
    printf("Test Accuracy: %.2f%%\n", accuracy);

    /* Cleanup */
    free(x);
    free(z1);
    free(a1);
    free(grad_a1);
    free(grad_z1);
    dense_free(&l1);
}

/* ========== Process 1: Hidden Layer 2 ========== */
void process_1_run(int num_train_samples, int num_test_samples) {
    DenseLayer l2;
    dense_init(&l2, HIDDEN1_SIZE, HIDDEN2_SIZE);

    /* Buffers */
    float *a1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *z2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *a2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *grad_a2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *grad_z2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *grad_a1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));

    MPI_Status status;

    printf("[Process 1] Ready for training\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double train_start = MPI_Wtime();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < num_train_samples; i++) {
            /* Receive activation from Process 0 */
            MPI_Recv(a1, HIDDEN1_SIZE, MPI_FLOAT, 0, TAG_FORWARD_A1, MPI_COMM_WORLD, &status);

            /* Forward: Dense Layer 2 + ReLU */
            dense_forward(&l2, a1, z2);
            relu_forward(z2, a2, HIDDEN2_SIZE);

            /* Send activation to Process 2 */
            MPI_Send(a2, HIDDEN2_SIZE, MPI_FLOAT, 2, TAG_FORWARD_A2, MPI_COMM_WORLD);

            /* Receive gradient from Process 2 */
            MPI_Recv(grad_a2, HIDDEN2_SIZE, MPI_FLOAT, 2, TAG_BACKWARD_GRAD2, MPI_COMM_WORLD, &status);

            /* Backward */
            relu_backward(z2, grad_a2, grad_z2, HIDDEN2_SIZE);
            dense_backward(&l2, grad_z2, grad_a1);

            /* Send gradient to Process 0 */
            MPI_Send(grad_a1, HIDDEN1_SIZE, MPI_FLOAT, 0, TAG_BACKWARD_GRAD1, MPI_COMM_WORLD);

            /* Update weights */
            sgd_update(&l2, LEARNING_RATE);
        }
    }

    double train_elapsed = MPI_Wtime() - train_start;
    double max_train_elapsed;

    MPI_Reduce(&train_elapsed, &max_train_elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ========== Evaluation ========== */
    printf("[Process 1] Ready for evaluation\n");

    for (int i = 0; i < num_test_samples; i++) {
        MPI_Recv(a1, HIDDEN1_SIZE, MPI_FLOAT, 0, TAG_FORWARD_A1, MPI_COMM_WORLD, &status);

        dense_forward(&l2, a1, z2);
        relu_forward(z2, a2, HIDDEN2_SIZE);

        MPI_Send(a2, HIDDEN2_SIZE, MPI_FLOAT, 2, TAG_FORWARD_A2, MPI_COMM_WORLD);
    }

    /* Cleanup */
    free(a1);
    free(z2);
    free(a2);
    free(grad_a2);
    free(grad_z2);
    free(grad_a1);
    dense_free(&l2);
}

/* ========== Process 2: Output Layer + Loss ========== */
void process_2_run(int num_train_samples, int num_test_samples) {
    DenseLayer l3;
    dense_init(&l3, HIDDEN2_SIZE, NUM_CLASSES);

    /* Buffers */
    float *a2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *z3 = (float *)malloc(NUM_CLASSES * sizeof(float));
    float *probs = (float *)malloc(NUM_CLASSES * sizeof(float));
    float *grad_z3 = (float *)malloc(NUM_CLASSES * sizeof(float));
    float *grad_a2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));

    MPI_Status status;

    printf("[Process 2] Ready for training\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double train_start = MPI_Wtime();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int i = 0; i < num_train_samples; i++) {
            /* Receive activation from Process 1 */
            MPI_Recv(a2, HIDDEN2_SIZE, MPI_FLOAT, 1, TAG_FORWARD_A2, MPI_COMM_WORLD, &status);

            /* Receive label from Process 0 */
            int label;
            MPI_Recv(&label, 1, MPI_INT, 0, TAG_FORWARD_LABEL, MPI_COMM_WORLD, &status);

            /* Forward: Dense Layer 3 + Softmax */
            dense_forward(&l3, a2, z3);
            softmax(z3, probs, NUM_CLASSES);

            /* Calculate loss */
            total_loss += cross_entropy_loss(probs, label);

            /* Backward */
            cross_entropy_backward(probs, label, grad_z3, NUM_CLASSES);
            dense_backward(&l3, grad_z3, grad_a2);

            /* Send gradient to Process 1 */
            MPI_Send(grad_a2, HIDDEN2_SIZE, MPI_FLOAT, 1, TAG_BACKWARD_GRAD2, MPI_COMM_WORLD);

            /* Update weights */
            sgd_update(&l3, LEARNING_RATE);
        }

        /* Send average loss to Process 0 */
        float avg_loss = total_loss / num_train_samples;
        MPI_Send(&avg_loss, 1, MPI_FLOAT, 0, TAG_LOSS, MPI_COMM_WORLD);
    }

    double train_elapsed = MPI_Wtime() - train_start;
    double max_train_elapsed;

    MPI_Reduce(&train_elapsed, &max_train_elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ========== Evaluation ========== */
    printf("[Process 2] Ready for evaluation\n");

    int correct = 0;
    for (int i = 0; i < num_test_samples; i++) {
        MPI_Recv(a2, HIDDEN2_SIZE, MPI_FLOAT, 1, TAG_FORWARD_A2, MPI_COMM_WORLD, &status);

        int label;
        MPI_Recv(&label, 1, MPI_INT, 0, TAG_FORWARD_LABEL, MPI_COMM_WORLD, &status);

        dense_forward(&l3, a2, z3);
        softmax(z3, probs, NUM_CLASSES);

        /* Find predicted class */
        int pred = 0;
        float max_prob = probs[0];
        for (int k = 1; k < NUM_CLASSES; k++) {
            if (probs[k] > max_prob) {
                max_prob = probs[k];
                pred = k;
            }
        }

        if (pred == label) correct++;
    }

    /* Send accuracy to Process 0 */
    float accuracy = 100.0f * correct / num_test_samples;
    MPI_Send(&accuracy, 1, MPI_FLOAT, 0, TAG_LOSS, MPI_COMM_WORLD);

    /* Cleanup */
    free(a2);
    free(z3);
    free(probs);
    free(grad_z3);
    free(grad_a2);
    dense_free(&l3);
}

#include "pipeline_batch.h"

/* ========== Main Function ========== */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_PROCESSES) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly %d MPI processes.\n", NUM_PROCESSES);
            fprintf(stderr, "Usage: mpirun -np %d %s\n", NUM_PROCESSES, argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    srand(42);
    int skip = rank == 0 ? 0 : INPUT_SIZE * HIDDEN1_SIZE;
    if (rank == 2) skip += HIDDEN1_SIZE * HIDDEN2_SIZE;
    for (int i = 0; i < skip; i++) (void)rand();

    Dataset train = {0}, test = {0};
    int num_train_samples = 0, num_test_samples = 0;

    /* Only Process 0 loads the full dataset */
    if (rank == 0) {
        printf("==============================================\n");
        printf("Pipeline Parallelism Neural Network with MPI\n");
        printf("==============================================\n");
        printf("Architecture: %d -> %d -> %d -> %d\n", INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_CLASSES);
        printf("Process 0: Input + Hidden Layer 1\n");
        printf("Process 1: Hidden Layer 2\n");
        printf("Process 2: Output Layer + Loss\n");
        printf("==============================================\n\n");

        load_mnist(&train, &test);

        if (train.images == NULL || train.labels == NULL ||
            test.images == NULL || test.labels == NULL) {
            fprintf(stderr, "Error: Failed to load MNIST dataset.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        num_train_samples = train.num_samples;
        num_test_samples = test.num_samples;
        const char *limit = getenv("TRAIN_LIMIT");
        if (limit && atoi(limit) > 0 && atoi(limit) < num_train_samples)
            num_train_samples = atoi(limit);
        limit = getenv("TEST_LIMIT");
        if (limit && atoi(limit) > 0 && atoi(limit) < num_test_samples)
            num_test_samples = atoi(limit);
        test.num_samples = num_test_samples;

        printf("Loaded %d training samples and %d test samples\n\n", num_train_samples, num_test_samples);
    }

    /* Broadcast dataset sizes to all processes */
    MPI_Bcast(&num_train_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_test_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    /* Run process-specific functions */
    pipeline_batch_run(rank, &train, &test, num_train_samples, num_test_samples);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nTotal runtime (initialization + training + evaluation): %.2f seconds\n", end_time - start_time);
        free_dataset(&train);
        free_dataset(&test);
    }

    MPI_Finalize();
    return 0;
}
