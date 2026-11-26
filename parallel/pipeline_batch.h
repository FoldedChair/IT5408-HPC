/* Batch-synchronous GPipe-style schedule: forward all micro-batches, then
 * backward all micro-batches. Weights remain fixed until the batch ends. */
#define BATCH_SIZE 32
#define MICRO_SIZE 8

static void pipeline_batch_run(int rank, Dataset *train, Dataset *test,
                               int ntrain, int ntest) {
    int in = rank == 0 ? INPUT_SIZE : (rank == 1 ? HIDDEN1_SIZE : HIDDEN2_SIZE);
    int out = rank == 0 ? HIDDEN1_SIZE : (rank == 1 ? HIDDEN2_SIZE : NUM_CLASSES);
    DenseLayer layer;
    dense_init(&layer, in, out);
    float *inputs = calloc(BATCH_SIZE * in, sizeof(float));
    float *z = calloc(BATCH_SIZE * out, sizeof(float));
    float *activations = calloc(BATCH_SIZE * out, sizeof(float));
    float *grad_out = calloc(BATCH_SIZE * out, sizeof(float));
    float *grad_in = calloc(BATCH_SIZE * in, sizeof(float));
    int labels[BATCH_SIZE];
    MPI_Request requests[2 * BATCH_SIZE / MICRO_SIZE];
    const char *epoch_env = getenv("TRAIN_EPOCHS");
    int epochs = epoch_env && atoi(epoch_env) > 0 ? atoi(epoch_env) : EPOCHS;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_start = MPI_Wtime();
        float loss = 0.0f;
        for (int base = 0; base < ntrain; base += BATCH_SIZE) {
            int count = ntrain - base < BATCH_SIZE ? ntrain - base : BATCH_SIZE;
            int nr = 0;
            tensor_zero(&layer.grad_w);
            tensor_zero(&layer.grad_b);
            for (int m = 0; m < count; m += MICRO_SIZE) {
                int n = count - m < MICRO_SIZE ? count - m : MICRO_SIZE;
                if (rank == 0) {
                    memcpy(inputs + m * in, train->images + (base + m) * in, n * in * sizeof(float));
                    for (int j = 0; j < n; j++) labels[m + j] = train->labels[base + m + j];
                    MPI_Isend(labels + m, n, MPI_INT, 2, TAG_FORWARD_LABEL, MPI_COMM_WORLD, &requests[nr++]);
                } else {
                    MPI_Recv(inputs + m * in, n * in, MPI_FLOAT, rank - 1,
                             rank == 1 ? TAG_FORWARD_A1 : TAG_FORWARD_A2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (rank == 2)
                        MPI_Recv(labels + m, n, MPI_INT, 0, TAG_FORWARD_LABEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                for (int j = m; j < m + n; j++) {
                    dense_forward(&layer, inputs + j * in, z + j * out);
                    if (rank < 2) relu_forward(z + j * out, activations + j * out, out);
                    else {
                        softmax(z + j * out, activations + j * out, out);
                        loss += cross_entropy_loss(activations + j * out, labels[j]);
                        cross_entropy_backward(activations + j * out, labels[j], grad_out + j * out, out);
                    }
                }
                if (rank < 2)
                    MPI_Isend(activations + m * out, n * out, MPI_FLOAT, rank + 1,
                              rank == 0 ? TAG_FORWARD_A1 : TAG_FORWARD_A2, MPI_COMM_WORLD, &requests[nr++]);
            }
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            nr = 0;
            for (int m = 0; m < count; m += MICRO_SIZE) {
                int n = count - m < MICRO_SIZE ? count - m : MICRO_SIZE;
                if (rank < 2)
                    MPI_Recv(grad_out + m * out, n * out, MPI_FLOAT, rank + 1,
                             rank == 0 ? TAG_BACKWARD_GRAD1 : TAG_BACKWARD_GRAD2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = m; j < m + n; j++) {
                    if (rank < 2) relu_backward(z + j * out, grad_out + j * out, grad_out + j * out, out);
                    memcpy(layer.input_cache.data, inputs + j * in, in * sizeof(float));
                    dense_backward(&layer, grad_out + j * out, rank ? grad_in + j * in : NULL);
                }
                if (rank > 0)
                    MPI_Isend(grad_in + m * in, n * in, MPI_FLOAT, rank - 1,
                              rank == 1 ? TAG_BACKWARD_GRAD1 : TAG_BACKWARD_GRAD2, MPI_COMM_WORLD, &requests[nr++]);
            }
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            sgd_update(&layer, LEARNING_RATE / count);
        }
        if (rank == 2) MPI_Send(&loss, 1, MPI_FLOAT, 0, TAG_LOSS, MPI_COMM_WORLD);
        if (rank == 0) {
            MPI_Recv(&loss, 1, MPI_FLOAT, 2, TAG_LOSS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Epoch %d | Loss: %.6f | Time: %.2f s\n", epoch + 1, loss / ntrain, MPI_Wtime() - epoch_start);
        }
    }
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("MPI Training time: %.2f seconds\n", max_elapsed);
    const char *dump_prefix = getenv("DUMP_PARAMETERS");
    if (dump_prefix) {
        char path[1024];
        snprintf(path, sizeof(path), "%s%d.bin", dump_prefix, rank + 1);
        FILE *file = fopen(path, "wb");
        if (!file) { perror(path); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fwrite(layer.weights.data, sizeof(float), (size_t)in * out, file) != (size_t)in * out ||
            fwrite(layer.bias.data, sizeof(float), out, file) != (size_t)out || fclose(file)) {
            fprintf(stderr, "Cannot write %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (getenv("CHECK_WEIGHTS")) {
        double values[2] = {0.0, 0.0}, all[6];
        for (int j = 0; j < in * out; j++) {
            double v = layer.weights.data[j];
            values[0] += v; values[1] += v * v;
        }
        MPI_Gather(values, 2, MPI_DOUBLE, all, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int k = 0; k < 3; k++)
                printf("Layer %d weights: %.12g %.12g\n", k + 1, all[2*k], all[2*k+1]);
    }

    int correct = 0;
    for (int i = 0; i < ntest; i++) {
        MPI_Request label_request;
        if (rank == 0) {
            memcpy(inputs, test->images + i * in, in * sizeof(float));
            labels[0] = test->labels[i];
            MPI_Isend(labels, 1, MPI_INT, 2, TAG_FORWARD_LABEL, MPI_COMM_WORLD, &label_request);
        } else {
            MPI_Recv(inputs, in, MPI_FLOAT, rank - 1, rank == 1 ? TAG_FORWARD_A1 : TAG_FORWARD_A2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank == 2) MPI_Recv(labels, 1, MPI_INT, 0, TAG_FORWARD_LABEL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        dense_forward(&layer, inputs, z);
        if (rank < 2) {
            relu_forward(z, activations, out);
            MPI_Send(activations, out, MPI_FLOAT, rank + 1, rank == 0 ? TAG_FORWARD_A1 : TAG_FORWARD_A2, MPI_COMM_WORLD);
            if (rank == 0) MPI_Wait(&label_request, MPI_STATUS_IGNORE);
        } else {
            softmax(z, activations, out);
            int pred = 0;
            for (int j = 1; j < out; j++) if (activations[j] > activations[pred]) pred = j;
            correct += pred == labels[0];
        }
    }
    if (rank == 2) MPI_Send(&correct, 1, MPI_INT, 0, TAG_LOSS, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&correct, 1, MPI_INT, 2, TAG_LOSS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Test Accuracy: %.2f%%\n", 100.0f * correct / ntest);
    }
    free(inputs); free(z); free(activations); free(grad_out); free(grad_in);
    dense_free(&layer);
}
