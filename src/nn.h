/**
 * ml - a neural network processor written with C
 * Copyright (C) 2023  jvech
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __NN__
#define __NN__

#include <stdbool.h>
#include <stddef.h>

struct Cost {
    double (*func)(double labels[], double net_out[], size_t shape);
    double (*dfunc_out)(double labels, double net_out);
};

struct Activation {
    double (*func)(double);
    double (*dfunc)(double);
};

typedef struct Layer {
    double *weights, *bias;
    struct Activation activation;
    size_t neurons, input_nodes;
} Layer;

void nn_network_write_weights(char *filepath, Layer *network, size_t network_size);
void nn_network_read_weights(char *filepath, Layer *network, size_t network_size);
void nn_network_init_weights(Layer *network, size_t nmemb, size_t input_cols, bool fill_random);
void nn_network_free_weights(Layer *network, size_t nmemb);

void nn_network_predict(
        double *out, size_t out_shape[2],
        double *input, size_t input_shape[2],
        Layer network[], size_t network_size);

void nn_network_train(
        Layer network[], size_t network_size,
        double *input, size_t input_shape[2],
        double *labels, size_t labels_shape[2],
        struct Cost cost, size_t epochs,
        size_t batch_size, double alpha,
        bool shuffle);

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
        double (*activation_derivative)(double));

void nn_layer_hidden_delta(
        double *delta, double *delta_next, double *zout,
        double *weights_next, size_t weights_next_shape[2],
        double (*activation_derivative)(double));
#endif
