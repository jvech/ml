#ifndef PARSE_H
#define PARSE_H

#include <stdio.h>
#include <stdbool.h>

#include "util.h"

enum ArrayType {
    ARRAY_NUMERICAL,
    ARRAY_ORDINAL,
    ARRAY_ONEHOT
};

union ArrayValue {
    double numeric;
    char *categorical;
};

typedef struct Array {
    enum ArrayType *type;
    union ArrayValue *data;
    size_t shape[2];
} Array;


void array_free(Array *x);
void file_read(char *filepath, Array *input, Array *out, struct Configs configs, bool read_output);
void file_write(Array input, Array out, struct Configs ml_configs);
char * file_format_infer(char *filename);
double * data_preprocess(
        size_t out_shape[2],
        Array data,
        struct Configs configs,
        bool is_input,
        bool only_allocate);

void data_postprocess(
        Array *out,
        double *data, size_t data_shape[2],
        struct Configs cfgs,
        bool is_input);
#endif
