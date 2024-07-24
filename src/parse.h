#ifndef PARSE_H
#define PARSE_H

#include <stdio.h>
#include <stdbool.h>

typedef struct Array {
    double *data;
    size_t shape[2];
} Array;


void file_read(
        char *filepath,
        Array *input, Array *out,
        char *in_keys[], size_t n_in_keys,
        char *out_keys[], size_t n_out_keys,
        bool read_output,
        char *file_format
        );
#endif
