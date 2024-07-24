#ifndef PARSE_H
#define PARSE_H

#include <stdio.h>
#include <stdbool.h>

typedef struct Array {
    double *data;
    size_t shape[2];
} Array;

void json_read(
        FILE *fp,
        Array *input, Array *out,
        char *out_keys[], size_t out_keys_size,
        char *in_keys[], size_t in_keys_size,
        bool read_output
        );

void csv_read(
        FILE *fp,
        Array *input, Array *out,
        char *in_cols[], size_t in_cols_size,
        char *out_cols[], size_t out_cols_size,
        bool read_output,
        bool has_header,
        char separator
        );
#endif
