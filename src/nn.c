#include "nn.h"

static void fill_random_weights(double *weights, double *bias, size_t rows, size_t cols);

void nn_layer_init_weights(Layer layer[], size_t nmemb, size_t n_inputs)
{
    int i;
    size_t prev_size = n_inputs;


    for (i = 0; i < nmemb;  i++) {
        layer[i].weights = calloc(prev_size * layer[i].neurons, sizeof(Layer));
        layer[i].bias = calloc(prev_size, sizeof(Layer));

        if (layer[i].weights == NULL || layer[i].bias == NULL) {
            goto nn_layer_calloc_weights_error;
        }
        fill_random_weights(layer[i].weights, layer[i].bias, prev_size, layer[i].neurons);
        prev_size = layer[i].neurons;
    }

    return;

nn_layer_calloc_weights_error:
    perror("nn_layer_calloc_weights() Error");
    exit(1);
}

void nn_layer_free_weights(Layer *layer, size_t nmemb)
{
    for (int i = 0; i < nmemb; i++) {
        free(layer[i].weights);
    }
}

double * nn_layer_forward(Layer layer, double *input, size_t input_shape[2])
{
    double *out = NULL;
    return out;
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
    int64_t *random_bias = calloc(rows, sizeof(int64_t));

    fread(random_weights, sizeof(double), weights_size, fp);
    fread(random_bias, sizeof(double), rows, fp);

    if (!random_weights || !random_bias) goto nn_fill_random_weights_error;

    for (size_t i = 0; i < weights_size; i++) {
        weights[i] = (double)random_weights[i] / (double)INT64_MAX * 2;
    }

    for (size_t i = 0; i < weights_size; i++) {
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
