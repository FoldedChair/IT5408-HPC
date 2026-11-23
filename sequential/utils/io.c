#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t read_u32(FILE *f)
{
    uint8_t b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

float *read_idx_images(const char *path, int *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

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

uint8_t *read_idx_labels(const char *path, int *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    read_u32(f);   /* magic */
    *count = read_u32(f);

    uint8_t *labels = (uint8_t *)malloc(*count);
    fread(labels, 1, *count, f);

    fclose(f);
    return labels;
}
