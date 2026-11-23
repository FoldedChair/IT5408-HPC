#include "math_utils.h"
#include <stdlib.h>

float rand_uniform(float min, float max)
{
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}
