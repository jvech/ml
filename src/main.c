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
    die("load_config('%s') Error:", filepath);
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


int main(int argc, char *argv[]) {
    char default_config_path[512], *env_config_path;
    struct Configs ml_configs = {
        .epochs = 100,
        .batch_size = 32,
        .alpha = 1e-5,
        .shuffle = true,
        .config_filepath = "",
        .network_size = 0,
        .only_out = false,
        .decimal_precision = -1,
        .file_format = NULL,
        .out_filepath = NULL,
    };

    // First past to check if --config option was put
    util_load_cli(&ml_configs, argc, argv);
    optind = 1;

    // Load configs with different possible paths
    sprintf(default_config_path, "%s/%s", getenv("HOME"), ".config/ml/ml.cfg");
    env_config_path = (getenv("ML_CONFIG_PATH"))? getenv("ML_CONFIG_PATH"):"";

    load_config(&ml_configs, 3,
                ml_configs.config_filepath,
                env_config_path,
                default_config_path);

    // re-read cli options again, to overwrite file configuration options
    util_load_cli(&ml_configs, argc, argv);
    argc -= optind;
    argv += optind;

    Layer *network = load_network(ml_configs);
    Array in, out;
    double *X = NULL, *y = NULL;
    size_t X_shape[2], y_shape[2];
    if (!strcmp("train", argv[0]) || !strcmp("retrain", argv[0])) {
        file_read(argv[1], &in, &out, ml_configs, true);
        X = data_preprocess(X_shape, in, ml_configs, true, false);
        y = data_preprocess(y_shape, out, ml_configs, false, false);
        if (!strcmp("train", argv[0])) {
            nn_network_init_weights(network, ml_configs.network_size, X_shape[1], true);
        } else if (!strcmp("retrain", argv[0])) {
            nn_network_init_weights(network, ml_configs.network_size, X_shape[1], false);
            nn_network_read_weights(ml_configs.weights_filepath, network, ml_configs.network_size);
        }
        nn_network_train(network, ml_configs, X, X_shape, y, y_shape);
        nn_network_write_weights(ml_configs.weights_filepath, network, ml_configs.network_size);
        fprintf(stderr, "weights saved on '%s'\n", ml_configs.weights_filepath);
    } else if (!strcmp("predict", argv[0])) {
        file_read(argv[1], &in, &out, ml_configs, false);
        X = data_preprocess(X_shape, in, ml_configs, true, false);
        y = data_preprocess(y_shape, out, ml_configs, false, true);
        nn_network_init_weights(network, ml_configs.network_size, X_shape[1], false);
        nn_network_read_weights(ml_configs.weights_filepath, network, ml_configs.network_size);
        nn_network_predict(y, y_shape, X, X_shape, network, ml_configs.network_size);

        // If neither output and file_format defined use input to define the output format
        if (!ml_configs.file_format && !ml_configs.out_filepath) {
            ml_configs.file_format = file_format_infer(ml_configs.in_filepath);
        }
        data_postprocess(&out, y, y_shape, ml_configs, false);
        file_write(in, out, ml_configs);
    } else usage(1);

    nn_network_free_weights(network, ml_configs.network_size);
    free(network);
    array_free(&in);
    array_free(&out);
    free(X);
    free(y);
    util_free_config(&ml_configs);
    return 0;
}
