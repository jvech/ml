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

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <getopt.h>
#include <json-c/json.h>

#include "util.h"
#include "parse.h"
#include "nn.h"

#define MAX_FILE_SIZE 536870912 //1<<29; 0.5 GiB

#define ARRAY_SIZE(x, type) sizeof(x) / sizeof(type)


static void json_write(
        const char *filepath,
        Array input, Array out,
        char *out_keys[], size_t out_keys_size,
        char *in_keys[], size_t in_keys_size);

void json_write(
        const char *filepath,
        Array input, Array out,
        char *out_keys[], size_t out_keys_size,
        char *in_keys[], size_t in_keys_size)
{
    FILE *fp = (!filepath) ? fopen("/dev/stdout", "w") : fopen(filepath, "w");
    if (!fp) die("json_read() Error:");
    fprintf(fp, "[\n");

    for (size_t i = 0; i < input.shape[0]; i++) {
        fprintf(fp, "  {\n");

        for (size_t j = 0; j < input.shape[1]; j++) {
            size_t index = input.shape[1] * i + j;
            fprintf(fp, "    \"%s\": %lf,\n", in_keys[j], input.data[index]);
        }

        for (size_t j = 0; j < out.shape[1]; j++) {
            size_t index = out.shape[1] * i + j;
            fprintf(fp, "    \"%s\": %lf", out_keys[j], out.data[index]);

            if (j == out.shape[1] - 1) fprintf(fp, "\n");
            else fprintf(fp, ",\n");
        }

        if (i == input.shape[0] - 1) fprintf(fp, "  }\n");
        else fprintf(fp, "  },\n");
    }
    fprintf(fp, "]\n");
    fclose(fp);
}

void load_config(struct Configs *cfg, int n_args, ...)
{
    char *filepath;
    va_list ap;
    va_start(ap, n_args);
    int i;
    for (i = 0; i < n_args; i++) {
        filepath = va_arg(ap, char *);
        util_load_config(cfg, filepath);
        if (errno == 0) {
            va_end(ap);
            return;
        } else if (errno == ENOENT && i < n_args - 1) {
            errno = 0;
        } else break;
    }
    va_end(ap);
    die("load_config() Error:");
}

Layer * load_network(struct Configs cfg)
{
    extern struct Activation NN_RELU;
    extern struct Activation NN_SOFTPLUS;
    extern struct Activation NN_SIGMOID;
    extern struct Activation NN_LEAKY_RELU;
    extern struct Activation NN_LINEAR;
    extern struct Activation NN_TANH;

    Layer *network = ecalloc(cfg.network_size, sizeof(Layer));

    for (size_t i = 0; i < cfg.network_size; i++) {
        if (!strcmp("relu", cfg.activations[i]))                network[i].activation = NN_RELU;
        else if (!strcmp("sigmoid", cfg.activations[i]))        network[i].activation = NN_SIGMOID;
        else if (!strcmp("softplus", cfg.activations[i]))       network[i].activation = NN_SOFTPLUS;
        else if (!strcmp("leaky_relu", cfg.activations[i]))     network[i].activation = NN_LEAKY_RELU;
        else if (!strcmp("linear", cfg.activations[i]))         network[i].activation = NN_LINEAR;
        else if (!strcmp("tanh", cfg.activations[i]))           network[i].activation = NN_TANH;
        else die("load_network() Error: Unknown '%s' activation", cfg.activations[i]);

        network[i].neurons = cfg.neurons[i];
    }
    return network;
}

struct Cost load_loss(struct Configs cfg)
{
    extern struct Cost NN_SQUARE;
    if (!strcmp("square", cfg.loss)) return NN_SQUARE;
    die("load_loss() Error: Unknown '%s' loss function", cfg.loss);
    exit(1);
}

int main(int argc, char *argv[]) {
    char default_config_path[512];
    struct Configs ml_configs = {
        .epochs = 100,
        .alpha = 1e-5,
        .config_filepath = "utils/settings.cfg",
        .network_size = 0,
        .file_format = NULL,
        .out_filepath = NULL,
    };

    // First past to check if --config option was put
    util_load_cli(&ml_configs, argc, argv);
    optind = 1;
    // Load configs with different possible paths
    sprintf(default_config_path, "%s/%s", getenv("HOME"), ".config/ml/ml.cfg");
    load_config(&ml_configs, 2, ml_configs.config_filepath, default_config_path);

    // re-read cli options again, to overwrite file configuration options
    util_load_cli(&ml_configs, argc, argv);
    argc -= optind;
    argv += optind;

    Layer *network = load_network(ml_configs);

    Array X, y;
    if (!strcmp("train", argv[0])) {
        file_read(argv[1], &X, &y, ml_configs.input_keys, ml_configs.n_input_keys, ml_configs.label_keys, ml_configs.n_label_keys, true, ml_configs.file_format);
        nn_network_init_weights(network, ml_configs.network_size, X.shape[1], true);
        nn_network_train(
                network, ml_configs.network_size,
                X.data, X.shape,
                y.data, y.shape,
                load_loss(ml_configs),
                ml_configs.epochs,
                ml_configs.alpha);
        nn_network_write_weights(ml_configs.weights_filepath, network, ml_configs.network_size);
        fprintf(stderr, "weights saved on '%s'\n", ml_configs.weights_filepath);
    } else if (!strcmp("predict", argv[0])) {
        file_read(argv[1], &X, &y, ml_configs.input_keys, ml_configs.n_input_keys, ml_configs.label_keys, ml_configs.n_label_keys, false, ml_configs.file_format);
        nn_network_init_weights(network, ml_configs.network_size, X.shape[1], false);
        nn_network_read_weights(ml_configs.weights_filepath, network, ml_configs.network_size);
        nn_network_predict(y.data, y.shape, X.data, X.shape, network, ml_configs.network_size);
        json_write(ml_configs.out_filepath, X, y, ml_configs.label_keys, ml_configs.n_label_keys, ml_configs.input_keys, ml_configs.n_input_keys);
    } else usage(1);

    nn_network_free_weights(network, ml_configs.network_size);
    free(network);
    util_free_config(&ml_configs);
    return 0;
}
