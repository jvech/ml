#include "nn.h"

static void fill_random_weights(double *weights, double *bias, size_t rows, size_t cols);

void nn_layer_backward(
        double *weights, size_t weigths_shape[2],
        double *delta, size_t delta_cols,
        double *out_prev, size_t out_cols,
        Layer layer, double alpha)
{
    assert(out_cols == weigths_shape[0] && "out_cols does not match with weight rows");
    assert(delta_cols == weigths_shape[1] && "delta_cols does not match with weight cols");

    for (size_t i = 0; i < weigths_shape[0]; i++) {
        for (size_t j = 0; j < weigths_shape[0]; j++) {
            size_t index = weigths_shape[1] * i + j;
            double dcost_w = delta[j] * out_prev[i];
            weights[index] = layer.weights[index] + alpha * dcost_w;
        }
    }
}

void nn_layer_hidden_delta(
        double *delta, size_t delta_cols,
        double *delta_next, size_t delta_next_cols,
        double *weigths_next, size_t weigths_shape[2],
        double *zout, size_t zout_cols,
        double (*activation_derivative)(double))
{
    assert(delta_cols == zout_cols);
    assert(delta_cols == weigths_shape[0]);
    assert(delta_next_cols == weigths_shape[1]);

    for (size_t j = 0; j < delta_cols; j++) {
        double sum = 0;
        for (size_t k = 0; k < delta_next_cols; k++) {
            size_t index = j * delta_cols + k;
            sum += delta_next[k] * weigths_next[index];
        }
        delta[j] = sum * activation_derivative(zout[j]);
    }
}

void nn_layer_out_delta(
        double *delta, size_t delta_cols,
        double *error, size_t error_cols,
        double *zout, size_t zout_cols,
        double (*activation_derivative)(double))
{
    assert(delta_cols == error_cols);
    assert(zout_cols == error_cols);

    for (size_t i = 0; i < delta_cols; i++) {
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
        nn_layer_map_activation(network[l].activation, out[l], out_shape, zout[l], out_shape);
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
