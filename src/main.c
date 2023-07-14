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
Layer neural[] = {
    {.neurons = 5, .activation = relu},
    {.neurons = 1, .activation = sigmoid},
};

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
    FILE *fp;
    char *fp_buffer;
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
    Array X, y;
    char *in_keys[] = {"area", "longitude", "latitude"};
    json_read("data/test.json", &X, &y, "price", in_keys, ARRAY_SIZE(in_keys, char *));

    nn_layer_init_weights(neural, ARRAY_SIZE(neural, Layer), X.shape[1]);
    double *out = nn_layer_forward(neural[0], X.data, X.shape);

    printf("area\tlat\tlong\t| price\n");
    for (size_t i = 0; i < X.shape[0]; i++) {
        for (size_t j = 0; j < X.shape[1]; j++) {
            size_t index = X.shape[1] * i + j;
            printf("%.2lf\t", X.data[index]);
        }
        printf("| %.2lf\n", y.data[i]);
    }

    printf("---\n");
    for (size_t i = 0; i < X.shape[0]; i++) {
        for (size_t j = 0; j < neural[0].neurons; j++) {
            size_t index = neural[0].neurons * i + j;
            printf("%.2lf\t", out[index]);
        }
        printf("\n");
    }


    nn_layer_free_weights(neural, ARRAY_SIZE(neural, Layer));
    free(out);
    free(X.data);
    free(y.data);
}
