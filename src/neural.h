#ifndef __NEURAL__
#define __NEURAL__

#include <stdlib.h>
#include <stdint.h>

typedef struct Layer {
    double *weights;
    double (*activation)(double x);
    size_t neurons;
} Layer;
#endif
