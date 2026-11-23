#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define PARAM_SIZE 4
#define MICRO_BATCHES 4

void draw_pipeline_diagram(int size);

int main(int argc, char **argv) {
    int rank, size;
    double data[PARAM_SIZE] = {1.0, 2.0, 3.0, 4.0};
    double recv_data[PARAM_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Need at least 2 MPI processes\n");
        MPI_Finalize();
        return 0;
    }

    // Forward pass: each process receives from prev, computes, sends to next
    for (int mb = 0; mb < MICRO_BATCHES; mb++) {
        if (rank > 0) {
            MPI_Recv(recv_data, PARAM_SIZE, MPI_DOUBLE, rank - 1, mb,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < PARAM_SIZE; i++) data[i] = recv_data[i] * 1.1;
        }
        if (rank < size - 1) {
            MPI_Send(data, PARAM_SIZE, MPI_DOUBLE, rank + 1, mb, MPI_COMM_WORLD);
        }
    }

    // Backward pass: gradients flow in reverse
    for (int mb = MICRO_BATCHES - 1; mb >= 0; mb--) {
        if (rank < size - 1) {
            MPI_Recv(recv_data, PARAM_SIZE, MPI_DOUBLE, rank + 1, mb + MICRO_BATCHES,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank > 0) {
            for (int i = 0; i < PARAM_SIZE; i++) recv_data[i] = data[i] * 0.1;
            MPI_Send(recv_data, PARAM_SIZE, MPI_DOUBLE, rank - 1, mb + MICRO_BATCHES,
                     MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        draw_pipeline_diagram(size);
    }

    MPI_Finalize();
    return 0;
}

void draw_pipeline_diagram(int size) {
    printf("\n================ PIPELINE PARALLELISM =================\n\n");
    printf("Pipeline Structure\n");

    // Draw boxes
    for (int i = 0; i < size; i++) {
        printf("┌─────────────┐");
        if (i < size - 1) printf("        ");
    }
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("│  Process %-2d │", i);
        if (i < size - 1) printf("───────>");
    }
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("│  Layer %-2d   │", i + 1);
        if (i < size - 1) printf("  A^(%d) ", i + 1);
    }
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("└─────────────┘");
        if (i < size - 1) printf("        ");
    }
    printf("\n");

    // Draw gradient arrows - box width 15 + gap 8 = 23 chars per unit
    printf("     ^");
    for (int i = 1; i < size - 1; i++) printf("                      ^");
    printf("                      │\n");
    printf("     │");
    for (int i = 1; i < size - 1; i++) printf("       gradients      │");
    printf("       gradients      │\n");
    printf("     └");
    for (int i = 1; i < size - 1; i++) printf("──────────────────────┴");
    printf("──────────────────────┘\n");

    // Micro-batching timeline
    printf("\nMicro-batching to Reduce Bubble Time\n\n");
    printf("Time");
    for (int i = 0; i < size; i++) printf("   Process %-2d", i);
    printf("\n");

    int total_steps = MICRO_BATCHES + size - 1 + size;
    for (int t = 1; t <= total_steps; t++) {
        printf("%3d ", t);
        for (int p = 0; p < size; p++) {
            int mb = t - p;
            if (mb >= 1 && mb <= MICRO_BATCHES) {
                printf("   micro-b%-3d", mb);
            } else if (t > MICRO_BATCHES + p && t <= total_steps - (size - 1 - p)) {
                printf("   backward  ");
            } else {
                printf("   idle      ");
            }
        }
        printf("\n");
    }
}
