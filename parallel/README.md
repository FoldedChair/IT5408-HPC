# Pipeline Parallelism Neural Network with MPI

This implementation demonstrates **pipeline parallelism** for training a fully connected neural network on the MNIST dataset using MPI (Message Passing Interface).

## Architecture

The neural network has the following architecture:
- **Input**: 784 neurons (28x28 MNIST images)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

## Pipeline Parallelism Design

Each MPI process handles specific layers of the neural network:

| Process | Responsibility |
|---------|----------------|
| **Process 0** | Input handling + Hidden Layer 1 (784 → 256) |
| **Process 1** | Hidden Layer 2 (256 → 128) |
| **Process 2** | Output Layer + Loss calculation (128 → 10) |

### Data Flow

**Forward Pass:**
```
Process 0: Input → Dense1 → ReLU → [send a1 to P1, send label to P2]
Process 1: [recv a1 from P0] → Dense2 → ReLU → [send a2 to P2]
Process 2: [recv a2 from P1, recv label from P0] → Dense3 → Softmax → Loss
```

**Backward Pass:**
```
Process 2: Loss Gradient → Dense3 Backward → [send grad_a2 to P1]
Process 1: [recv grad_a2 from P2] → ReLU Backward → Dense2 Backward → [send grad_a1 to P0]
Process 0: [recv grad_a1 from P1] → ReLU Backward → Dense1 Backward
```

## Building

```bash
make
```

## Running

Run with exactly 3 MPI processes:

```bash
make run
# or
mpirun -np 3 ./fcn_mpi
```

If you have fewer than 3 CPU cores, use the oversubscribe option:

```bash
make run-oversubscribe
# or
mpirun -np 3 --oversubscribe ./fcn_mpi
```

## Dataset

Place the MNIST dataset files in the `data/` folder:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Configuration

You can modify the following parameters in `fcn_mpi.c`:

```c
#define INPUT_SIZE     784
#define HIDDEN1_SIZE   256
#define HIDDEN2_SIZE   128
#define NUM_CLASSES    10

#define EPOCHS         10
#define LEARNING_RATE  0.01f
```

## Output Example

```
==============================================
Pipeline Parallelism Neural Network with MPI
==============================================
Architecture: 784 -> 256 -> 128 -> 10
Process 0: Input + Hidden Layer 1
Process 1: Hidden Layer 2
Process 2: Output Layer + Loss
==============================================

Loaded 60000 training samples and 10000 test samples

[Process 0] Starting training with 60000 samples
[Process 1] Ready for training
[Process 2] Ready for training
Epoch 1 | Loss: 0.5432 | Time: 12.34 s
...
Test Accuracy: 87.50%

Total execution time: 125.67 seconds
```

## Clean

```bash
make clean
```

---

## Legacy Guide

```shell
# Install lib
brew install open-mpi
mpicc --showme:incdirs

# Compile
mpicc -o fcn_mpi_0_1 fcn_mpi_0_1.c

# Run with 4 process
mpirun -np 4 ./fcn_mpi_0_1

# Run
mpicc -o fcn_mpi_0_1 fcn_mpi_0_1.c && mpirun -np 4 ./fcn_mpi_0_1
```


