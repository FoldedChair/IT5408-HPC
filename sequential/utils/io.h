#ifndef IO_H
#define IO_H

#include <stdint.h>

float   *read_idx_images(const char *path, int *count);
uint8_t *read_idx_labels(const char *path, int *count);

#endif
