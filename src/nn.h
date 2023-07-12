#ifndef __NN__
#define __NN__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

typedef struct Layer {
    double *weights, *bias;
    double (*activation)(double x);
    size_t neurons, input_size;
} Layer;

void nn_layer_init_weights(Layer *layer, size_t nmemb, size_t input_size);
void nn_layer_free_weights(Layer *layer, size_t nmemb);

double * nn_layer_forward(Layer layer, double *input, size_t input_shape[2]);
double * nn_layer_backward(Layer layer, double *output, size_t out_shape[2]);

double sigmoid(double x);
double relu(double x);
#endif
