CC = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic
INCLUDES = -I.

SRC = \
main.c \
data/mnist_loader.c \
nn/tensor.c \
nn/layer_dense.c \
nn/activations.c \
nn/loss.c \
nn/optimizer_sgd.c \
nn/network.c \
utils/math_utils.c \
utils/memory.c \
utils/io.c

OBJ = $(SRC:.c=.o)

all: digits_nn

digits_nn: $(OBJ)
	$(CC) $(CFLAGS) -o digits_nn $(OBJ) -lm

clean:
	rm -f $(OBJ) digits_nn
