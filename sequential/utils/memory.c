#include "memory.h"
#include <stdlib.h>

float *alloc_f32(size_t n)
{
    return (float *)malloc(sizeof(float) * n);
}

void safe_free(void *p)
{
    if (p) free(p);
}
