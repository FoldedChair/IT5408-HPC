# Sequential Neural Network

Sequential implementation of a fully connected neural network for MNIST classification. This version serves as the optimized sequential baseline for comparison with the MPI parallel implementation.

## Architecture

```text
Input: 784 neurons (28x28 grayscale image)
        |
Dense Layer 1: 784 -> 256
        |
ReLU
        |
Dense Layer 2: 256 -> 128
        |
ReLU
        |
Output Layer: 128 -> 10
        |
Softmax
        |
Cross-Entropy Loss
```

## Configuration

| Parameter | Value |
|---|---|
| Input size | 784 |
| Hidden layer 1 | 256 |
| Hidden layer 2 | 128 |
| Output size | 10 |
| Learning rate | 0.01 |
| Epochs | 10 |

## Requirements

- Linux or WSL Ubuntu
- GCC
- GNU Make
- Standard C libraries
- Math library (`libm`)

No MPI, Python, or external machine-learning libraries are required for the sequential implementation.

## Installation

Update package lists:

```bash
sudo apt update
```

Install the required build tools and utilities:

```bash
sudo apt install -y build-essential gcc make openmpi-bin libopenmpi-dev
```

Verify the installation:

```bash
gcc --version
make --version
```

## Dataset Setup

This project uses the MNIST Handwritten digits dataset. The dataset is included in the repository. 

Expected directory structure:

```text
high_performance_computing_project/
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── sequential/
│   ├── Makefile
│   ├── main.c
│   └── ...
└── parallel/
    └── ...
```

The sequential implementation expects the dataset files to be available in the project's `data/` directory.

## Build

Navigate to the sequential directory from the project root:

```bash
cd sequential/
```

Clean previous build artifacts:

```bash
make clean
```

Build the executable:

```bash
make
```

The Makefile uses the following compiler flags:

```bash
-std=c99 -O3 -Wall -Wextra -pedantic
```

The build produces the executable:

```text
digits_nn
```

## Run

From the `sequential/` directory:

```bash
./digits_nn
```

The program trains the neural network for 10 epochs and reports:

- Loss per epoch
- Epoch training time
- Total training time
- Test accuracy

## Clean Build

To remove compiled object files and the executable:

```bash
make clean
```

To clean and rebuild:

```bash
make clean && make
```
