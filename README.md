# Model Parallelism for MNIST Classification

This project trains a fully connected neural network in C using a sequential baseline and a three-stage MPI pipeline. The dataset in `main` is MNIST handwritten digits.

## Model and training

The network has three trainable dense layers:

```text
784 input pixels -> 256 ReLU -> 128 ReLU -> 10 Softmax outputs
```

Images are 28 x 28 pixels, flattened and normalized to [0, 1]. The included IDX dataset has 60,000 training images and 10,000 test images. Both implementations use mini-batch SGD, a batch size of 32, learning rate 0.01, 10 epochs and random seed 42 by default. Loss is cross-entropy and evaluation reports test accuracy.

The MPI program requires exactly three processes:

| Rank | Stage |
| --- | --- |
| 0 | Input and dense layer 1 |
| 1 | Dense layer 2 |
| 2 | Dense layer 3, loss and prediction |

Each batch is divided into micro-batches of eight. The pipeline performs all forward passes and then all backward passes, accumulating gradients before each batch update. Activations and activation gradients pass between adjacent ranks; labels pass from rank 0 to rank 2. Each rank owns its layer's parameters.

## Requirements

Use Linux or WSL with GCC, GNU Make, Open MPI and the C math library. For example, on Ubuntu:

```sh
sudo apt install build-essential openmpi-bin libopenmpi-dev
```

The programs expect these files in `data/`:

```text
data/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

## Build and run

Run from the project root. The executables expect their working directories to be `sequential/` and `parallel/`, respectively, so that `../data/` resolves correctly.

```sh
make -C sequential
(cd sequential && ./digits_nn)

make -C parallel
(cd parallel && mpirun -np 3 ./fcn_mpi)
```

To clean build products:

```sh
make -C sequential clean
make -C parallel clean
```

For a short smoke run, both implementations support the following environment variables:

```sh
(cd sequential && TRAIN_LIMIT=1000 TEST_LIMIT=100 TRAIN_EPOCHS=1 ./digits_nn)
(cd parallel && TRAIN_LIMIT=1000 TEST_LIMIT=100 TRAIN_EPOCHS=1 mpirun -np 3 ./fcn_mpi)
```

`TRAIN_LIMIT` and `TEST_LIMIT` restrict the number of samples; `TRAIN_EPOCHS` overrides the default epoch count. These are intended for quick checks, not the full benchmark.

## Benchmark

```sh
bash benchmark/run.sh
```

The script builds both programs, runs three sequential and three MPI trials in alternating order, and writes logs and environment information under a timestamped `benchmark/results-*` directory. It records dataset SHA-256 checksums. Run on an otherwise idle machine and record the source revision separately: the script does not record a Git commit or control CPU affinity.

The historical MNIST benchmark in `benchmark/results-20261126-183608/` reports median training times of 130.91 s sequential and 76.89 s MPI, or 1.70x; both reported 95.64% test accuracy. These results describe those runs, not guaranteed performance on another machine or revision. The sequential implementation in `main` computes an unused first-layer input gradient that the MPI implementation skips, so the speedup does not isolate parallelism alone.

The older `benchmark/results-20261126-163819/` logs correspond to a different historical dataset; the current `data/` directory contains MNIST handwritten digits.

## Repository layout

- `sequential/`: sequential training and evaluation implementation.
- `parallel/`: MPI implementation and micro-batch pipeline.
- `data/`: MNIST IDX files.
- `benchmark/`: benchmark script and historical run logs.
