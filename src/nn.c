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

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <openblas/cblas.h>

#include "util.h"
#include "nn.h"

static void fill_random_weights(double *weights, double *bias, size_t rows, size_t cols);
static double get_avg_loss(
        double labels[], double outs[], size_t shape[2],
        double (*loss)(double *, double *, size_t));


double square_loss(double labels[], double net_outs[], size_t shape);
double square_dloss_out(double labels, double net_out);

struct Cost NN_SQUARE = {
    .func = square_loss,
    .dfunc_out = square_dloss_out
};

void nn_network_predict(
        double *output, size_t output_shape[2],
        double *input, size_t input_shape[2],
        Layer network[], size_t network_size)
{
    double **outs = calloc(network_size, sizeof(double *));
    double **zouts = calloc(network_size, sizeof(double *));
    size_t samples = input_shape[0];
    for (size_t l = 0; l < network_size; l++) {
        outs[l] = calloc(samples * network[l].neurons, sizeof(double));
        zouts[l] = calloc(samples * network[l].neurons, sizeof(double));
    }

    nn_forward(outs, zouts, input, input_shape, network, network_size);
    memmove(output, outs[network_size - 1], samples * output_shape[1] * sizeof(double));

    for (size_t l = 0; l < network_size; l++) {
        free(outs[l]);
        free(zouts[l]);
    }
    free(outs);
    free(zouts);
}

void nn_network_train(
        Layer network[], size_t network_size,
        double *input, size_t input_shape[2],
        double *labels, size_t labels_shape[2],
        struct Cost cost, size_t epochs,
        size_t batch_size, double alpha)
{
    assert(input_shape[0] == labels_shape[0] && "label samples don't correspond with input samples\n");

    double **outs = calloc(network_size, sizeof(double *));
    double **zouts = calloc(network_size, sizeof(double *));
    double **weights = calloc(network_size, sizeof(double *));
    double **biases = calloc(network_size, sizeof(double *));

    if (!outs || !zouts || !weights || !biases) goto nn_network_train_error;



    size_t samples = input_shape[0];
    for (size_t l = 0; l < network_size; l++) {
        outs[l] = calloc(batch_size * network[l].neurons, sizeof(double));
        zouts[l] = calloc(batch_size * network[l].neurons, sizeof(double));
        weights[l] = malloc(network[l].input_nodes * network[l].neurons * sizeof(double));
        biases[l] = malloc(network[l].neurons * sizeof(double));

        if (!outs[l] || !zouts || !weights[l] || !biases) goto nn_network_train_error;


        memcpy(weights[l], network[l].weights, sizeof(double) * network[l].input_nodes * network[l].neurons);
        memcpy(biases[l], network[l].bias, sizeof(double) * network[l].neurons);
    }


    size_t batch_input_shape[2] = {batch_size, input_shape[1]};
    size_t batch_labels_shape[2] = {batch_size, labels_shape[1]};
    size_t n_batches = input_shape[0] / batch_size;
    if (samples % batch_size) {
        n_batches++;
    }
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        for (size_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
            size_t index = batch_size * batch_idx;

            double *input_batch = input + index * input_shape[1];
            double *labels_batch = labels + index * labels_shape[1];

            if (batch_idx == n_batches - 1 && samples % batch_size) {
                batch_input_shape[0] = samples % batch_size;
                batch_labels_shape[0] = samples % batch_size;
            }

            nn_forward(outs, zouts, input_batch, batch_input_shape, network, network_size);
            nn_backward(
                    weights, biases,
                    zouts, outs,
                    input_batch, batch_input_shape,
                    labels_batch, batch_labels_shape,
                    network, network_size,
                    cost.dfunc_out, alpha);
            double *net_out = outs[network_size - 1];
            fprintf(stdout, "epoch: %g \t loss: %6.6lf\n",
                    epoch + (float)batch_idx / n_batches,
                    get_avg_loss(labels, net_out, batch_labels_shape, cost.func));
        }
    }

    for (size_t l = 0; l < network_size; l++) {
        free(outs[l]);
        free(zouts[l]);
        free(weights[l]);
        free(biases[l]);
    }

    free(zouts);
    free(outs);
    free(weights);
    free(biases);

    return;
nn_network_train_error:
    perror("nn_network_train() Error");
    exit(1);
}

void nn_backward(
        double **weights, double **bias,
        double **Zout, double **Outs,
        double *Input, size_t input_shape[2],
        double *Labels, size_t labels_shape[2],
        Layer network[], size_t network_size,
        double (dcost_out_func)(double, double),
        double alpha)
{
    size_t max_neurons = 0;
    for (size_t l = 0; l < network_size; l++) {
        max_neurons = (max_neurons > network[l].neurons) ? max_neurons : network[l].neurons;
    }
    double *dcost_outs = calloc(labels_shape[0] * labels_shape[1], sizeof(double));
    double *delta = calloc(max_neurons, sizeof(double));
    double *delta_next = calloc(max_neurons, sizeof(double));

    if (!dcost_outs || !delta || !delta_next) goto nn_backward_error;

    for (size_t i = 0; i < labels_shape[0]; i++) {
        for (size_t j = 0; j < labels_shape[1]; j++) {
            size_t index = i * labels_shape[1] + j;
            dcost_outs[index] = dcost_out_func(Labels[index], Outs[network_size - 1][index]);
        }
    }

    for (size_t sample = 0; sample < input_shape[0]; sample++) {
        for (size_t l = network_size - 1; l < network_size; l--) {
            size_t weights_shape[2] = {network[l].input_nodes, network[l].neurons};
            if (l == network_size - 1) {
                double *zout = Zout[l] + sample * network[l].neurons;
                double *out_prev = Outs[l - 1] + sample * network[l-1].neurons;
                double *dcost_out = dcost_outs + sample * network[l].neurons;
                nn_layer_out_delta(delta, dcost_out, zout, network[l].neurons, network[l].activation.dfunc);
                nn_layer_backward(weights[l], bias[l], weights_shape, delta, out_prev, network[l], alpha);
            } else if (l == 0) {
                size_t weights_next_shape[2] = {network[l+1].input_nodes, network[l+1].neurons};
                double *zout = Zout[l] + sample * network[l].neurons;
                double *input = Input + sample * input_shape[1];
                nn_layer_hidden_delta(delta, delta_next, zout, weights[l+1], weights_next_shape, network[l].activation.dfunc);
                nn_layer_backward(weights[l], bias[l], weights_shape, delta, input, network[l], alpha);
            } else {
                size_t weights_next_shape[2] = {network[l+1].input_nodes, network[l+1].neurons};
                double *zout = Zout[l] + sample * network[l].neurons;
                double *out_prev = Outs[l - 1] + sample * network[l-1].neurons;
                nn_layer_hidden_delta(delta, delta_next, zout, weights[l+1], weights_next_shape, network[l].activation.dfunc);
                nn_layer_backward(weights[l], bias[l], weights_shape, delta, out_prev, network[l], alpha);
            }
            memmove(delta_next, delta, weights_shape[1] * sizeof(double));
        }

    }

    for (size_t l = 0; l < network_size; l++) {
        size_t weights_shape[2] = {network[l].input_nodes, network[l].neurons};
        memcpy(network[l].weights, weights[l], weights_shape[0] * weights_shape[1] * sizeof(double));
        memcpy(network[l].bias, bias[l], weights_shape[1] * sizeof(double));
    }

    free(dcost_outs);
    free(delta);
    free(delta_next);
    return;
nn_backward_error:
    perror("nn_backward() Error");
    exit(1);
}

void nn_layer_backward(
        double *weights, double *bias, size_t weights_shape[2],
        double *delta, double *out_prev,
        Layer layer, double alpha)
{
    // W_next = W - alpha * out_prev @ delta.T
    cblas_dger(CblasRowMajor, weights_shape[0], weights_shape[1], -alpha,
               out_prev, 1, delta, 1, weights, weights_shape[1]);

    for (size_t j = 0; j < weights_shape[1]; j++)
        bias[j] = bias[j] - alpha * delta[j];
}

void nn_layer_hidden_delta(
        double *delta, double *delta_next, double *zout,
        double *weights_next, size_t weights_shape[2],
        double (*activation_derivative)(double))
{
    for (size_t j = 0; j < weights_shape[0]; j++) {
        double sum = 0;
        for (size_t k = 0; k < weights_shape[1]; k++) {
            size_t index = j * weights_shape[1] + k;
            sum += delta_next[k] * weights_next[index];
        }
        delta[j] = sum * activation_derivative(zout[j]);
    }
}

void nn_layer_out_delta(
        double *delta, double *error, double *zout,
        size_t cols,
        double (*activation_derivative)(double))
{

    for (size_t i = 0; i < cols; i++) {
        delta[i] = error[i] * activation_derivative(zout[i]);
    }
}

void nn_forward(
        double **out, double **zout,
        double *X, size_t X_shape[2],
        Layer network[], size_t network_size)
{
    size_t in_shape[2] = {X_shape[0], X_shape[1]};
    size_t out_shape[2];
    out_shape[0] = X_shape[0];
    double *input = X;

    for (size_t l = 0; l < network_size; l++) {
        out_shape[1] = network[l].neurons;
        nn_layer_forward(network[l], zout[l], out_shape, input, in_shape);
        nn_layer_map_activation(network[l].activation.func, out[l], out_shape, zout[l], out_shape);
        in_shape[1] = out_shape[1];
        input = out[l];
    }
}

void nn_layer_map_activation(
        double (*activation)(double),
        double *aout, size_t aout_shape[2],
        double *zout, size_t zout_shape[2])
{
    if (zout_shape[0] != aout_shape[0] || zout_shape[1] != aout_shape[1]) {
        fprintf(stderr,
                "nn_layer_map_activation() Error: zout must have (%zu x %zu) dimensions not (%zu x %zu)\n",
                aout_shape[0], aout_shape[1], zout_shape[0], zout_shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < aout_shape[0]; i++) {
        for (size_t j = 0; j < aout_shape[1]; j ++) {
            size_t index = aout_shape[1] * i + j;
            aout[index] = activation(zout[index]);
        }
    }
}

void nn_layer_forward(Layer layer, double *zout, size_t zout_shape[2], double *input, size_t input_shape[2])
{
    if (zout_shape[0] != input_shape[0] || zout_shape[1] != layer.neurons) {
        fprintf(stderr,
                "nn_layer_forward() Error: zout must have (%zu x %zu) dimensions not (%zu x %zu)\n",
                input_shape[0], layer.neurons, zout_shape[0], zout_shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < input_shape[0]; i++) {
        for (size_t j = 0; j < layer.neurons; j++) {
            size_t index = layer.neurons * i + j;
            zout[index] = layer.bias[j];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                input_shape[0], layer.neurons, layer.input_nodes, // m, n, k
                1.0, input, input_shape[1], //alpha X
                layer.weights, layer.neurons, // W
                1.0, zout, layer.neurons); // beta B
}

void nn_network_read_weights(char *filepath, Layer *network, size_t network_size)
{
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL) die("nn_network_read_weights Error():");

    size_t net_size, shape[2], ret;
    ret = fread(&net_size, sizeof(size_t), 1, fp);
    if (net_size != network_size) goto nn_network_read_weights_error;

    for (size_t i = 0; i < network_size; i++) {
        fread(shape, sizeof(size_t), 2, fp);
        if (shape[0] != network[i].input_nodes
            || shape[1] != network[i].neurons) {
            goto nn_network_read_weights_error;
        }

        if (!network[i].weights || !network[i].bias) {
            die("nn_network_read_weights() Error: "
                "the weights on layer %zu haven't been initialized", i);
        }

        ret = fread(network[i].weights, sizeof(double), shape[0] * shape[1], fp);
        if (ret != shape[0] * shape[1]) goto nn_network_read_weights_error;

        ret = fread(network[i].bias, sizeof(double), shape[1], fp);
        if (ret != shape[1]) goto nn_network_read_weights_error;
    }

    fclose(fp);
    return;

nn_network_read_weights_error:
    fclose(fp);
    die("nn_network_read_weights() Error: "
        "number of read objects does not match with expected ones");
}

void nn_network_write_weights(char *filepath, Layer *network, size_t network_size)
{
    FILE *fp = fopen(filepath, "wb");
    if (fp == NULL) die("nn_network_write_weights() Error:");

    fwrite(&network_size, sizeof(size_t), 1, fp);

    size_t ret;
    for (size_t i = 0; i < network_size; i++) {
        size_t shape[2] = {network[i].input_nodes, network[i].neurons};
        size_t size = shape[0] * shape[1];

        ret = fwrite(shape, sizeof(size_t), 2, fp);
        if (ret != 2) goto nn_network_write_weights_error;

        ret = fwrite(network[i].weights, sizeof(double), size, fp);
        if (ret != size) goto nn_network_write_weights_error;

        ret = fwrite(network[i].bias, sizeof(double), network[i].neurons, fp);
        if (ret != network[i].neurons) goto nn_network_write_weights_error;
    }
    fclose(fp);
    return;

nn_network_write_weights_error:
    fclose(fp);
    die("nn_network_write_weights() Error: "
        "number of written objects does not match with number of objects");
}

void nn_network_init_weights(Layer layers[], size_t nmemb, size_t n_inputs, bool fill_random)
{
    size_t i, prev_size = n_inputs;


    for (i = 0; i < nmemb;  i++) {
        layers[i].weights = calloc(prev_size * layers[i].neurons, sizeof(double));
        layers[i].bias = calloc(layers[i].neurons, sizeof(double));

        if (layers[i].weights == NULL || layers[i].bias == NULL) {
            goto nn_layers_calloc_weights_error;
        }

        if (fill_random) fill_random_weights(layers[i].weights, layers[i].bias, prev_size, layers[i].neurons);

        layers[i].input_nodes = prev_size;
        prev_size = layers[i].neurons;
    }

    return;

nn_layers_calloc_weights_error:
    perror("nn_layers_calloc_weights() Error");
    exit(1);
}

void nn_network_free_weights(Layer layers[], size_t nmemb)
{
    size_t i;
    for (i = 0; i < nmemb; i++) {
        free(layers[i].weights);
        free(layers[i].bias);
    }
}

void fill_random_weights(double *weights, double *bias, size_t rows, size_t cols)
{
    FILE *fp = fopen("/dev/random", "rb");
    if (fp == NULL) goto nn_fill_random_weights_error;

    size_t weights_size = rows * cols;
    int64_t *random_weights = calloc(weights_size, sizeof(int64_t));
    int64_t *random_bias = calloc(cols, sizeof(int64_t));

    fread(random_weights, sizeof(int64_t), weights_size, fp);
    fread(random_bias, sizeof(int64_t), cols, fp);

    if (!random_weights || !random_bias) goto nn_fill_random_weights_error;

    for (size_t i = 0; i < weights_size; i++) {
        weights[i] = (double)random_weights[i] / (double)INT64_MAX * 2;
    }

    for (size_t i = 0; i < cols; i++) {
        bias[i] = (double)random_bias[i] / (double)INT64_MAX * 2;
    }

    free(random_weights);
    free(random_bias);
    fclose(fp);
    return;

nn_fill_random_weights_error:
    perror("nn_fill_random_weights Error()");
    exit(1);
}

double square_loss(double labels[], double net_out[], size_t shape)
{
    double sum = 0;
    for (size_t i = 0; i < shape; i++) {
        sum += pow(labels[i] - net_out[i], 2);
    }
    return 0.5 * sum;
}

double square_dloss_out(double label, double net_out)
{
    return net_out - label;
}

double get_avg_loss(
        double labels[], double outs[], size_t shape[2],
        double (*loss)(double *, double *, size_t shape))
{
    double sum = 0;
    for (size_t i = 0; i < shape[0]; i += shape[1]) {
        sum += loss(labels + i, outs + i, shape[1]);
    }
    return sum / shape[0];
}
