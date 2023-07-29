#ifndef __NN__
#define __NN__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <openblas/cblas.h>

typedef struct Layer {
    double *weights, *bias;
    double (*activation)(double x);
    size_t neurons, input_nodes;
} Layer;

void nn_network_init_weights(Layer *network, size_t nmemb, size_t input_cols);
void nn_network_free_weights(Layer *network, size_t nmemb);

void nn_layer_map_activation(double (*activation)(double), double *aout, size_t aout_shape[2], double *zout, size_t zout_shape[2]);
void nn_layer_forward(Layer layer, double *out, size_t out_shape[2], double *input, size_t input_shape[2]);
void nn_layer_backward(
        Layer *layer,
        double *weights,
        double *out, size_t out_shape[2],
        double *labels, size_t labels_shape[2],
        double *local_gradient); //TODO

double sigmoid(double x);
double relu(double x);
double identity(double x);


void nn_forward(double **aout, double **zout, double *input, size_t input_shape[2], Layer network[], size_t network_size);
double nn_layer_out_delta(double error, double (*activation_derivative)(double));
#endif
