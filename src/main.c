#include <stdio.h>
#include <stdint.h>
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


static void json_read(const char *filepath,
                      Array *input, Array *out,
                      char *out_key,
                      char *in_keys[],
                      size_t in_keys_size);

void json_read(const char *filepath,
               Array *input, Array *out,
               char *out_key,
               char *in_keys[],
               size_t n_input_keys)
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
    out->shape[1] = 1;
    out->data = calloc(out->shape[0] * out->shape[1], sizeof(out->data[0]));

    for (int i = 0; i < json_object_array_length(json_obj); i++) {
        json_object *item = json_object_array_get_idx(json_obj, i);

        out->data[i] = json_object_get_double(json_object_object_get(item, out_key));
        for (int j = 0; j < n_input_keys; j++) {
            input->data[n_input_keys * i + j] = json_object_get_double(json_object_object_get(item, in_keys[j]));
        }
    }

    json_object_put(json_obj);
    fclose(fp);

    return;

json_read_error:
    perror("json_read() Error");
    exit(1);
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


int main(int argc, char *argv[]) {
    struct Configs ml_configs = {
        .epochs = 100,
        .alpha = 1e-5,
        .config_filepath = "utils/settings.cfg",
        .network_size = 0,
    };

    // Try different config paths
    load_config(&ml_configs, 3, "~/.config/ml/ml.cfg", "~/.ml/ml.cfg", ml_configs.config_filepath);
    util_load_cli(&ml_configs, argc, argv);
    util_free_config(&ml_configs);
    return 0;
}
