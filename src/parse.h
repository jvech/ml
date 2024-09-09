#ifndef PARSE_H
#define PARSE_H

#include <stdio.h>
#include <stdbool.h>

#include "util.h"

typedef struct Array {
    double *data;
    size_t shape[2];
} Array;


void file_read(char *filepath, Array *input, Array *out, struct Configs configs, bool read_output);
void file_write(Array input, Array out, struct Configs ml_configs);
char * file_format_infer(char *filename);
#endif
