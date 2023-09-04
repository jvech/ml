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
#include "nn.h"

const size_t MAX_FILE_SIZE = 1<<29; // 0.5 GiB

typedef struct Array {
    double *data;
    size_t shape[2];
} Array;

#define ARRAY_SIZE(x, type) sizeof(x) / sizeof(type)


static void json_read(
        const char *filepath,
        Array *input, Array *out,
        char *out_keys[], size_t out_keys_size,
        char *in_keys[], size_t in_keys_size,
        bool read_output);

static void json_write(
        const char *filepath,
        Array input, Array out,
        char *out_keys[], size_t out_keys_size,
        char *in_keys[], size_t in_keys_size);

void json_read(
        const char *filepath,
        Array *input, Array *out,
        char *out_keys[], size_t n_out_keys,
        char *in_keys[], size_t n_input_keys,
        bool read_output)
{
    FILE *fp = NULL;
    char *fp_buffer = NULL;
    size_t ret;
    int64_t fp_size;

    fp = fopen(filepath, "r");
    if (fp == NULL) goto json_read_error;

    ret = (size_t)fseek(fp, 0L, SEEK_END);
    if ((int)ret == -1) goto json_read_error;

    fp_size = ftell(fp);
    if (fp_size == -1) goto json_read_error;
    if (fp_size >= MAX_FILE_SIZE) {
        fprintf(stderr, "ftell Error(): '%s' size greater than '%zu'\n", filepath, MAX_FILE_SIZE);
    }
    rewind(fp);

    fp_buffer = calloc(sizeof(char), fp_size);
    if (fp_buffer == NULL) goto json_read_error;

    ret = fread(fp_buffer, sizeof(char), (size_t)fp_size, fp);
    if (ret != (size_t)fp_size) {
        fprintf(stderr, "json_read() Error: fread bytes '%zd' does not match with buffer size '%zd'", ret, (size_t)fp_size);
        exit(1);
    }


    json_object *json_obj;
    json_obj = json_tokener_parse(fp_buffer);
    size_t json_obj_length = json_object_array_length(json_obj);

    input->shape[0] = (size_t)json_obj_length;
    input->shape[1] = n_input_keys;
    input->data = calloc(input->shape[0] * input->shape[1], sizeof(input->data[0]));

    out->shape[0] = (size_t)json_obj_length;
    out->shape[1] = n_out_keys;
    out->data = calloc(out->shape[0] * out->shape[1], sizeof(out->data[0]));

    if (!input->data || !out->data) goto json_read_error;

    for (int i = 0; i < json_object_array_length(json_obj); i++) {
        json_object *item = json_object_array_get_idx(json_obj, i);

        for (int j = 0; j < n_input_keys; j++) {
            size_t index = n_input_keys * i + j;
            input->data[index] = json_object_get_double(json_object_object_get(item, in_keys[j]));
        }

        if (!read_output) continue;

        for (int j = 0; j < n_out_keys; j++) {
            size_t index =  n_out_keys * i + j;
            out->data[index] = json_object_get_double(json_object_object_get(item, out_keys[j]));
        }
    }

    json_object_put(json_obj);
    fclose(fp);

    return;

json_read_error:
    perror("json_read() Error");
    exit(1);
}

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

    Layer *network = ecalloc(cfg.network_size, sizeof(Layer));

    for (size_t i = 0; i < cfg.network_size; i++) {
        if (!strcmp("relu", cfg.activations[i]))                network[i].activation = NN_RELU;
        else if (!strcmp("sigmoid", cfg.activations[i]))        network[i].activation = NN_SIGMOID;
        else if (!strcmp("softplus", cfg.activations[i]))       network[i].activation = NN_SOFTPLUS;
        else if (!strcmp("leaky_relu", cfg.activations[i]))     network[i].activation = NN_LEAKY_RELU;
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
    struct Configs ml_configs = {
        .epochs = 100,
        .alpha = 1e-5,
        .config_filepath = "utils/settings.cfg",
        .network_size = 0,
        .out_filepath = NULL,
    };

    // Try different config paths
    load_config(&ml_configs, 3, "~/.config/ml/ml.cfg", "~/.ml/ml.cfg", ml_configs.config_filepath);
    util_load_cli(&ml_configs, argc, argv);
    argc -= optind;
    argv += optind;
    Layer *network = load_network(ml_configs);

    Array X, y;
    if (!strcmp("train", argv[0])) {
        json_read(argv[1], &X, &y, ml_configs.label_keys, ml_configs.n_label_keys, ml_configs.input_keys, ml_configs.n_input_keys, true);
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
        json_read(argv[1], &X, &y, ml_configs.label_keys, ml_configs.n_label_keys, ml_configs.input_keys, ml_configs.n_input_keys, false);
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
