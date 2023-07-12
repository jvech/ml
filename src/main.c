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

Layer neural[] = {
    [0] = {.neurons = 3, .activation = relu},
    [1] = {.neurons = 1, .activation = sigmoid},
};

static Array json_read(const char *filepath);

Array json_read(const char *filepath)
{
    Array out;
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

    out.shape[0] = (size_t)json_obj_length;
    out.shape[1] = 4;
    out.data = calloc(out.shape[0] * out.shape[1], sizeof(out.data[0]));

    for (int i = 0; i < json_object_array_length(json_obj); i++) {
        json_object *item = json_object_array_get_idx(json_obj, i);
        out.data[4*i] = json_object_get_double(json_object_object_get(item, "area"));
        out.data[4*i + 1] = json_object_get_double(json_object_object_get(item, "longitude"));
        out.data[4*i + 2] = json_object_get_double(json_object_object_get(item, "latitude"));
        out.data[4*i + 3] = json_object_get_double(json_object_object_get(item, "price"));
    }

    json_object_put(json_obj);
    fclose(fp);
    return out;

json_read_error:
    perror("json_read() Error");
    exit(1);
}

int main(void) {
    Array json_data = json_read("data/housing_rent.json");
    nn_layer_init_weights(neural, 2, 3);
    printf("%lf\n", neural[0].weights[0]);
    nn_layer_free_weights(neural, 2);
    free(json_data.data);
    return 0;
}
