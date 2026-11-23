# Compiler
MPICC = mpicc

# Compiler flags
CFLAGS = -Wall -O3 -std=c99

# Libraries
LIBS = -lm

# Targets
TARGET = fcn_mpi
TARGET_PIPELINE = fcn_mpi_pipeline

# Source files
SRC = fcn_mpi.c

# Build main target
all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LIBS)

# Clean
clean:
	rm -f $(TARGET) $(TARGET_PIPELINE) *.o

# Run with 3 processes (pipeline parallelism)
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

# Run with oversubscribe option (if you have fewer than 3 CPU cores)
run-oversubscribe: $(TARGET)
	mpirun -np 3 --oversubscribe ./$(TARGET)

.PHONY: all clean run run-oversubscribe
