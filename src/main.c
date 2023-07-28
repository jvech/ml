#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <json-c/json.h>

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

int main(void) {
    Layer network[] = {
        {.neurons = 5, .activation = relu},
        {.neurons = 1, .activation = sigmoid},
    };
    Array X, y;
    char *in_keys[] = {"area", "longitude", "latitude"};
    json_read("data/test.json", &X, &y, "price", in_keys, ARRAY_SIZE(in_keys, char *));

    size_t network_size = ARRAY_SIZE(network, Layer);
    nn_network_init_weights(network, network_size, 3);
    double **outputs = calloc(network_size, sizeof(double *));

    size_t out_rows = X.shape[0];//
    for (size_t l = 0; l < network_size; l++) {//
        outputs[l] = calloc(network[l].neurons * out_rows, sizeof(double));//
    } //

    nn_forward(outputs, X.data, X.shape, network, network_size);

    for (size_t l = 0; l < network_size; l++) free(outputs[l]);
    free(outputs);

    nn_network_free_weights(network, network_size);
    free(X.data);
    free(y.data);

    
}
