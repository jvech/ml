#ifndef __NN__
#define __NN__

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <openblas/cblas.h>

typedef struct Layer {
    double *weights, *bias;
    double (*activation)(double x);
    double (*activation_derivative)(double x);
    size_t neurons, input_nodes;
} Layer;

void nn_network_init_weights(Layer *network, size_t nmemb, size_t input_cols);
void nn_network_free_weights(Layer *network, size_t nmemb);

void nn_layer_map_activation(
        double (*activation)(double),
        double *aout, size_t aout_shape[2],
        double *zout, size_t zout_shape[2]);


double sigmoid(double x);
double relu(double x);
double identity(double x);


void nn_forward(
        double **aout, double **zout,
        double *input, size_t input_shape[2],
        Layer network[], size_t network_size);

void nn_backward(
        double **weights, double **bias,
        double **zout, double **outs,
        double *input, size_t input_shape[2],
        double *labels, size_t labels_shape[2],
        Layer network[], size_t network_size,
        double (cost_derivative)(double, double),
        double alpha);

void nn_layer_forward(
        Layer layer,
        double *out, size_t out_shape[2],
        double *input, size_t input_shape[2]);

void nn_layer_backward(
        double *weights, double *bias, size_t weigths_shape[2],
        double *delta, double *out_prev,
        Layer layer, double alpha);

void nn_layer_out_delta(
        double *delta, double *dcost_out, double *zout, size_t cols,
        double (*activation_derivative)(double));//TODO

void nn_layer_hidden_delta(
        double *delta, double *delta_next, double *zout,
        double *weights_next, size_t weights_next_shape[2],
        double (*activation_derivative)(double));//TODO
#endif
