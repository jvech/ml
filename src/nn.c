#include "nn.h"

static void fill_random_weights(double *weights, double *bias, size_t rows, size_t cols);

void nn_forward(
        double **out,
        double *X, size_t X_shape[2],
        Layer network[], size_t network_size)
{
    size_t in_shape[2] = {X_shape[0], X_shape[1]};
    size_t out_shape[2];
    out_shape[0] = X_shape[0];
    double *input = X;

    for (size_t l = 0; l < network_size; l++) {
        out_shape[1] = network[l].neurons;
        nn_layer_forward(network[l], out[l], out_shape, input, in_shape);
        in_shape[1] = out_shape[1];
        input = out[l];
    }
}

void nn_layer_forward(Layer layer, double *out, size_t out_shape[2], double *input, size_t input_shape[2])
{
    if (out_shape[0] != input_shape[0] || out_shape[1] != layer.neurons) {
        fprintf(stderr,
                "nn_layer_forward() Error: out must have (%zu x %zu) dimensions not (%zu x %zu)\n",
                input_shape[0], layer.neurons, out_shape[0], out_shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < input_shape[0]; i++) {
        for (size_t j = 0; j < layer.neurons; j++) {
            size_t index = layer.neurons * i + j;
            out[index] = layer.bias[j];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                input_shape[0], layer.neurons, layer.input_nodes, // m, n, k
                1.0, input, input_shape[1], //alpha X
                layer.weights, layer.neurons, // W
                1.0, out, layer.neurons); // beta B

    for (size_t i = 0; i < input_shape[0]; i++) {
        for (size_t j = 0; j < layer.neurons; j ++) {
            size_t index = layer.neurons * i + j;
            out[index] = layer.activation(out[index]);
        }
    }
}

void nn_network_init_weights(Layer layers[], size_t nmemb, size_t n_inputs)
{
    int i;
    size_t prev_size = n_inputs;


    for (i = 0; i < nmemb;  i++) {
        layers[i].weights = calloc(prev_size * layers[i].neurons, sizeof(double));
        layers[i].bias = calloc(layers[i].neurons, sizeof(double));

        if (layers[i].weights == NULL || layers[i].bias == NULL) {
            goto nn_layers_calloc_weights_error;
        }
        fill_random_weights(layers[i].weights, layers[i].bias, prev_size, layers[i].neurons);
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
    for (int i = 0; i < nmemb; i++) {
        free(layers[i].weights);
        free(layers[i].bias);
    }
}

double identity(double x)
{
    return x;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double relu(double x)
{
    return (x > 0) ? x : 0;
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
