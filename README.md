# Specific Problem: Fully Connected Network for Classification

Below is a detailed guide for the **Fully Connected Network** problem using **Model Parallelism** with MPI:

## 1. Problem Specification

### Network Model

Fully Connected Neural Network (Multi-Layer Perceptron) with 4 layers:

- **Input layer**: 784 neurons (28×28 pixel images, e.g., MNIST)
- **Hidden layer 1**: 256 neurons + ReLU activation
- **Hidden layer 2**: 128 neurons + ReLU activation
- **Output layer**: 10 neurons + Softmax (classification of 10 classes: 0-9)


### Dataset

MNIST or Fashion-MNIST (60,000 training images, 10,000 testing images)[^1][^2]

### Model Parallelism Objective

Divide the model using **Pipeline Parallelism**: each MPI process handles one or more layers.[^3][^4]

```
Process 0: Input + Hidden Layer 1
Process 1: Hidden Layer 2
Process 2: Output Layer + Loss calculation
```


## 2. Mathematical Formulation

### Forward Propagation

For each layer $l$, the calculation formula is as follows:[^2][^5]

**Layer 1** (Process 0):

$$
Z^{(1)} = W^{(1)} \cdot X + b^{(1)}
$$

$$
A^{(1)} = \text{ReLU}(Z^{(1)}) = \max(0, Z^{(1)})
$$

Where:

- $W^{(1)}$: weight matrix of size $256 \times 784$
- $X$: input vector of size $784 \times 1$
- $b^{(1)}$: bias vector of size $256 \times 1$
- $A^{(1)}$: activation output sent to Process 1

**Layer 2** (Process 1):

$$
Z^{(2)} = W^{(2)} \cdot A^{(1)} + b^{(2)}
$$

$$
A^{(2)} = \text{ReLU}(Z^{(2)})
$$

**Output Layer** (Process 2):

$$
Z^{(3)} = W^{(3)} \cdot A^{(2)} + b^{(3)}
$$

$$
\hat{y} = \text{Softmax}(Z^{(3)}) = \frac{e^{Z^{(3)}_i}}{\sum_j e^{Z^{(3)}_j}}
$$

**Loss Function (Cross-Entropy)**:

$$
L = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)
$$

### Backward Propagation

**Gradients** calculated from output to input:[^6][^5]

**Layer 3** (Process 2):

$$
\frac{\partial L}{\partial Z^{(3)}} = \hat{y} - y
$$

$$
\frac{\partial L}{\partial W^{(3)}} = \frac{\partial L}{\partial Z^{(3)}} \cdot (A^{(2)})^T
$$

$$
\frac{\partial L}{\partial A^{(2)}} = (W^{(3)})^T \cdot \frac{\partial L}{\partial Z^{(3)}}
$$

**Layer 2** (Process 1):

$$
\frac{\partial L}{\partial Z^{(2)}} = \frac{\partial L}{\partial A^{(2)}} \odot \text{ReLU}'(Z^{(2)})
$$

Where $\text{ReLU}'(z) = 1$ if $z > 0$, otherwise 0.

$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial Z^{(2)}} \cdot (A^{(1)})^T
$$

**Layer 1** (Process 0): Similar to layer 2.

**Weight Update (Gradient Descent)**:

$$
W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}
$$

Where $\alpha$ is the learning rate (e.g., 0.01).

## 3. Parallel Algorithm Design

### Pipeline Structure

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Process 0  │─────>│  Process 1  │─────>│  Process 2  │
│  Layer 1    │ A^(1)│  Layer 2    │ A^(2)│  Layer 3    │
└─────────────┘      └─────────────┘      └─────────────┘
     ^                     ^                     │
     │        gradients    │        gradients    │
     └─────────────────────┴─────────────────────┘
```


### Micro-batching to Reduce Bubble Time

Instead of processing one sample at a time, divide a mini-batch (32 samples) into 4 micro-batches (8 samples per micro-batch):[^3]

```
Time  Process 0   Process 1   Process 2
  1   micro-b1    idle        idle
  2   micro-b2    micro-b1    idle
  3   micro-b3    micro-b2    micro-b1
  4   micro-b4    micro-b3    micro-b2
  5   backward    micro-b4    micro-b3
  6   backward    backward    micro-b4
  7   backward    backward    backward
```


## 4. MPI Pseudo-code

### Forward Propagation

```c
// Process 0: Layer 1
if (rank == 0) {
    // Read input data X[^784]
    matrix_multiply(W1, X, Z1);      // Z1 = W1 * X
    vector_add(Z1, b1, Z1);          // Z1 = Z1 + b1
    relu(Z1, A1);                    // A1 = ReLU(Z1)
    
    // Send A1 to Process 1
    MPI_Send(A1, 256, MPI_FLOAT, 1, TAG_FORWARD, MPI_COMM_WORLD);
}

// Process 1: Layer 2
else if (rank == 1) {
    // Receive A1 from Process 0
    MPI_Recv(A1, 256, MPI_FLOAT, 0, TAG_FORWARD, MPI_COMM_WORLD, &status);
    
    matrix_multiply(W2, A1, Z2);
    vector_add(Z2, b2, Z2);
    relu(Z2, A2);
    
    // Send A2 to Process 2
    MPI_Send(A2, 128, MPI_FLOAT, 2, TAG_FORWARD, MPI_COMM_WORLD);
}

// Process 2: Layer 3
else if (rank == 2) {
    MPI_Recv(A2, 128, MPI_FLOAT, 1, TAG_FORWARD, MPI_COMM_WORLD, &status);
    
    matrix_multiply(W3, A2, Z3);
    vector_add(Z3, b3, Z3);
    softmax(Z3, y_pred);             // y_pred = Softmax(Z3)
    
    // Calculate loss
    loss = cross_entropy(y_true, y_pred);
}
```


### Backward Propagation

```c
// Process 2: Backward through Layer 3
if (rank == 2) {
    // Loss gradient
    for (i = 0; i < 10; i++) {
        dL_dZ3[i] = y_pred[i] - y_true[i];
    }
    
    // Gradient of W3
    matrix_multiply(dL_dZ3, A2_transpose, dL_dW3);
    
    // Gradient sent back to Process 1
    matrix_multiply(W3_transpose, dL_dZ3, dL_dA2);
    MPI_Send(dL_dA2, 128, MPI_FLOAT, 1, TAG_BACKWARD, MPI_COMM_WORLD);
}

// Process 1: Backward through Layer 2
else if (rank == 1) {
    MPI_Recv(dL_dA2, 128, MPI_FLOAT, 2, TAG_BACKWARD, MPI_COMM_WORLD, &status);
    
    // Element-wise multiplication with ReLU derivative
    for (i = 0; i < 128; i++) {
        dL_dZ2[i] = dL_dA2[i] * (Z2[i] > 0 ? 1 : 0);
    }
    
    matrix_multiply(dL_dZ2, A1_transpose, dL_dW2);
    matrix_multiply(W2_transpose, dL_dZ2, dL_dA1);
    
    MPI_Send(dL_dA1, 256, MPI_FLOAT, 0, TAG_BACKWARD, MPI_COMM_WORLD);
}

// Process 0: Backward through Layer 1
else if (rank == 0) {
    MPI_Recv(dL_dA1, 256, MPI_FLOAT, 1, TAG_BACKWARD, MPI_COMM_WORLD, &status);
    // Similar to Process 1...
}
```


### Weight Update

```c
// Each process updates the weights of its managed layer
if (rank == 0) {
    // W1 = W1 - alpha * dL_dW1
    for (i = 0; i < 256; i++) {
        for (j = 0; j < 784; j++) {
            W1[i][j] -= LEARNING_RATE * dL_dW1[i][j];
        }
    }
}
```


## 5. Sample C/MPI Code

### Matrix multiplication function (most important)

```c
void matrix_vector_multiply(float** W, float* X, float* result, 
                            int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += W[i][j] * X[j];
        }
    }
}
```


### ReLU activation function

```c
void relu(float* Z, float* A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (Z[i] > 0) ? Z[i] : 0;
    }
}
```


### Softmax function

```c
void softmax(float* Z, float* y_pred, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        y_pred[i] = exp(Z[i]);
        sum += y_pred[i];
    }
    for (int i = 0; i < size; i++) {
        y_pred[i] /= sum;
    }
}
```


## 6. Data Structure

```c
#define INPUT_SIZE 784
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01

// Process 0 needs
float W1[HIDDEN1_SIZE][INPUT_SIZE];
float b1[HIDDEN1_SIZE];
float A1[HIDDEN1_SIZE];

// Process 1 needs
float W2[HIDDEN2_SIZE][HIDDEN1_SIZE];
float b2[HIDDEN2_SIZE];
float A2[HIDDEN2_SIZE];

// Process 2 needs
float W3[OUTPUT_SIZE][HIDDEN2_SIZE];
float b3[OUTPUT_SIZE];
float y_pred[OUTPUT_SIZE];
```


## 7. Performance Measurement

### Metrics to Collect

```c
double start_time, end_time;
double forward_time, backward_time, comm_time;

// Measure forward pass time
start_time = MPI_Wtime();
// ... forward propagation code ...
forward_time = MPI_Wtime() - start_time;

// Measure communication overhead
start_time = MPI_Wtime();
MPI_Send(...);
comm_time = MPI_Wtime() - start_time;
```


### Comparison with Sequential

| Metric | Sequential | Parallel (3 processes) | Speedup |
| :-- | :-- | :-- | :-- |
| Time/epoch | T_seq | T_par | T_seq/T_par |
| Memory/process | Full model | ~1/3 model | - |
| Throughput | samples/s | samples/s | - |

### Communication Cost Analysis

According to theory:[^7][^8]

- Communication cost for All-to-One (BitTorrent): $O(Lk + \frac{kp}{B})$
- With $k=3$ processes, $p$ = activation size (256, 128)
- Communication time: $O(p \log k)$[^7]


## 8. Report Structure

### Chapter 1: Problem (1-2 pages)

- Description of image classification problem
- Neural network architecture
- Why Model Parallelism is needed


### Chapter 2: Mathematical Model (2 pages)

- Complete forward/backward propagation formulas
- Loss function and gradient descent formulas
- Clear mathematical notation


### Chapter 3: Parallel Algorithm (2-3 pages)

- Pipeline parallelism diagram
- MPI pseudo-code
- Micro-batching methodology
- Communication strategy


### Chapter 4: Implementation & Results (3-4 pages)

- Implementation details (compiler, hardware)
- Training time comparison table
- Graphs: Speedup, Efficiency, Communication overhead
- Accuracy comparison (sequential vs parallel)


## 9. Files to Submit

```
Group1_FCN_Classification_MPI.zip
├── report.pdf
├── slides.pptx
├── sequential/
│   ├── fcn_sequential.c
│   └── Makefile
├── parallel/
│   ├── fcn_mpi.c
│   └── Makefile
├── data/
│   ├── train_data.csv
│   └── test_data.csv
└── README.txt
```


### Sample README.txt

```
# Compilation
Sequential version:
  gcc -o fcn_seq sequential/fcn_sequential.c -lm

Parallel version:
  mpicc -o fcn_mpi parallel/fcn_mpi.c -lm

# Execution
Sequential:
  ./fcn_seq data/train_data.csv

Parallel (3 processes):
  mpirun -np 3 fcn_mpi data/train_data.csv

# Expected output
Training accuracy: ~95%
Testing accuracy: ~94%
```


## 10. Implementation Tips

1. **Start simple**: Implement the sequential version first, test thoroughly
2. **Debug step by step**: Test forward pass before implementing backward pass
3. **Matrix multiplication**: This is the most time-consuming part, optimize carefully[^8][^9]
4. **Memory management**: Use dynamic allocation for large matrices
5. **MPI Tags**: Use different TAGs for forward/backward to avoid errors

With this guide, you have all the knowledge needed to complete the Model Parallelism project for a Fully Connected network.[^10][^11][^7]
<span style="display:none">[^12][^13][^14][^15][^16]</span>

<div align="center">⁂</div>

[^1]: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/

[^2]: https://www.datacamp.com/tutorial/forward-propagation-neural-networks

[^3]: https://www.deepspeed.ai/tutorials/pipeline/

[^4]: https://afmck.in/posts/2023-02-26-parallelism/

[^5]: https://www.adityaagrawal.net/blog/deep_learning/bprop_fc

[^6]: https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/

[^7]: Deep Learning - Parallel and Distributed.pdf

[^8]: https://iitgoa.ac.in/wp-content/uploads/MPI-Matrix-Multiplication.pdf

[^9]: https://www.linkedin.com/pulse/matrix-multiplication-using-mpi-semanto-mondal-vnrvf

[^10]: Hướng dẫn làm Middle Project.pdf

[^11]: Chapter-2-Parallel-Algorithm-Design.pptx.pdf

[^12]: Report - Master Project - VV Thieu.pdf

[^13]: Chapter-3-Shared-memory-Programming-OpenMP.pptx.pdf

[^14]: Chapter 1 - Parallel Architectures.pptx.pdf

[^15]: https://princeton-introml.github.io/files/ch11.pdf

[^16]: https://soulpageit.com/ai-glossary/fully-connected-neural-networks-explained/

