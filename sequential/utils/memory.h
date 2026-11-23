#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>

float *alloc_f32(size_t n);
void safe_free(void *p);

#endif
