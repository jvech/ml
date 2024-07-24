#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <json-c/json.h>

#include "util.h"
#include "parse.h"

#define MAX_FILE_SIZE 536870912 //1<<29; 0.5 GiB

static void json_read(
        FILE *fp,
        Array *input, Array *out,
        char *in_keys[], size_t in_keys_size,
        char *out_keys[], size_t out_keys_size,
        bool read_output
        );

static void csv_read(
        FILE *fp,
        Array *input, Array *out,
        char *in_cols[], size_t in_cols_size,
        char *out_cols[], size_t out_cols_size,
        bool read_output,
        bool has_header,
        char separator
        );

static void csv_columns_select(
        double *dst_row, double *src_row,
        size_t selected_cols[], size_t cols_size,
        size_t src_cols_number);

static void csv_readline_values(
        double *num_buffer, size_t num_buffer_length,
        char *line_buffer, size_t line_number,
        char separator);

static void csv_keys2cols(size_t cols[], char *keys[], size_t keys_size);

void file_read(
        char *filepath,
        Array *input, Array *out,
        char *in_keys[], size_t n_in_keys,
        char *out_keys[], size_t n_out_keys,
        bool read_output,
        char *file_format)
{
    FILE *fp;
    char *ptr;
    int i, string_length;

    fp = (!strcmp(filepath, "-")) ? fopen("/dev/stdin", "r") : fopen(filepath, "r");
    if (fp == NULL) die("file_read() Error:");

    if (file_format == NULL && !strcmp(filepath, "-")) {
        die("file_read() Error: on standard input the format must be defined");
    }

    if (file_format == NULL) {
        string_length = strlen(filepath);
        ptr = filepath + string_length;
        for (i = string_length; i > 0 && *ptr != '.'; ptr--, i--);
        if (*ptr != '.' || i == 0) die("file_read() Error: unable to infer %s format", filepath);
        file_format = ptr + 1;
    }

    if (!strcmp(file_format, "csv"))        csv_read(fp, input, out, in_keys, n_in_keys, out_keys, n_out_keys, read_output, false, ',');
    else if (!strcmp(file_format, "tsv"))   csv_read(fp, input, out, in_keys, n_in_keys, out_keys, n_out_keys, read_output, false, '\t');
    else if (!strcmp(file_format, "json"))  json_read(fp, input, out, in_keys, n_in_keys, out_keys, n_out_keys, read_output);
    else {
        die("file_read() Error: unable to parse %s files", file_format);
    }

    fclose(fp);
}

void json_read(
        FILE *fp,
        Array *input, Array *out,
        char *in_keys[], size_t n_input_keys,
        char *out_keys[], size_t n_out_keys,
        bool read_output)
{
    static char fp_buffer[MAX_FILE_SIZE];


    if (fp == NULL) goto json_read_error;

    size_t i = 0;
    do {
        if (i >= MAX_FILE_SIZE) die("json_read() Error: file size is bigger than '%zu'", i, MAX_FILE_SIZE);
        fp_buffer[i] = fgetc(fp);
    } while (fp_buffer[i++] != EOF);

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
    return;

json_read_error:
    perror("json_read() Error");
    exit(1);
}


void csv_read(
        FILE *fp,
        Array *input, Array *out,
        char *in_keys[], size_t n_in_cols,
        char *out_keys[], size_t n_out_cols,
        bool read_output,
        bool has_header, //TODO
        char separator)
{
    char line_buffer[1024];
    char *line_ptr;
    double *num_buffer;
    size_t line = 0, num_buffer_length = 1; 
    size_t *in_cols, *out_cols;

    if (fp == NULL) die("csv_read() Error:");

    in_cols = ecalloc(n_in_cols, sizeof(size_t));
    csv_keys2cols(in_cols, in_keys, n_in_cols);

    out_cols = ecalloc(n_out_cols, sizeof(size_t));
    csv_keys2cols(out_cols, out_keys, n_out_cols);

    input->shape[0] = 1;
    input->shape[1] = n_in_cols;
    input->data = ecalloc(input->shape[1], sizeof(double));

    out->shape[0] = 1;
    out->shape[1] = n_out_cols;
    out->data = ecalloc(out->shape[1], sizeof(double));

    fgets(line_buffer, 1024, fp);
    for (line_ptr = line_buffer; *line_ptr != '\0'; line_ptr++) {
        if (*line_ptr == separator) {
            num_buffer_length++;
        }
    }

    num_buffer = ecalloc(num_buffer_length, sizeof(double));

    csv_readline_values(num_buffer, num_buffer_length, line_buffer, 1, separator);
    csv_columns_select(input->data + line * input->shape[1], num_buffer, in_cols, n_in_cols, num_buffer_length);
    if (read_output) {
        csv_columns_select(out->data + line * out->shape[1], num_buffer, out_cols, n_out_cols, num_buffer_length);
    }

    for (line = 1; fgets(line_buffer, 1024, fp) != NULL; line++) {
        csv_readline_values(num_buffer, num_buffer_length, line_buffer, line+1, separator);

        input->shape[0]++;
        input->data = erealloc(input->data, input->shape[0] * input->shape[1] * sizeof(double));
        csv_columns_select(input->data + line * input->shape[1], num_buffer, in_cols, n_in_cols, num_buffer_length);

        out->shape[0]++;
        out->data = erealloc(out->data, out->shape[0] * out->shape[1] * sizeof(double));
        if (read_output) {
            csv_columns_select(out->data + line * out->shape[1], num_buffer, out_cols, n_out_cols, num_buffer_length);
        }
    }
    free(num_buffer);
    free(in_cols);
    free(out_cols);
    return;
}

void csv_columns_select(
        double *dst_row, double *src_row,
        size_t selected_cols[], size_t cols_size,
        size_t src_cols_number)
{
    size_t i, selected_col;
    for (i = 0; i < cols_size; i++) {

        if (selected_cols[i] == 0) {
            die("csv_columns_select() Error: invalid %zu column, use 1-indexing",
                selected_cols[i]);
        }

        selected_col = selected_cols[i] - 1;
        if (selected_col >= src_cols_number) {
            die("csv_columns_select() Error: "
                "selected column %zu outranges row columns size %zu",
                selected_cols[i], src_cols_number);
        }
        dst_row[i] = src_row[selected_col];
    }
}

void csv_readline_values(
        double *num_buffer, size_t num_buffer_length,
        char *line_buffer, size_t line_number,
        char separator)
{
    char *line_ptr;
    size_t col;
    int offset;

    for (col = 0, offset = 0, line_ptr = line_buffer;
         col < num_buffer_length
         && sscanf(line_ptr, "%lf%n", num_buffer+col, &offset) >= 1;
         line_ptr+=offset, col++) {
        // Checks 
        if (*(line_ptr + offset) == separator || *(line_ptr + offset) == '\n') {
            offset++;
        } else {
            die("csv_readline_values() Error: on line %zu format separator must be '%c' not '%c'",
                line_number, separator, *(line_ptr + offset));
        }
    }

    if (col < num_buffer_length && *line_ptr == '\0') {
        die("csv_readline_values() Error: line %zu seems to have less than %zu columns",
            line_number, num_buffer_length);
    } else if (col == num_buffer_length && *line_ptr != '\0') {
        die("csv_readline_values() Error: line %zu seems to have more than %zu columns",
            line_number, num_buffer_length);
    } else if (*line_ptr != '\0') {
        die("csv_readline_values() Error: "
            "line %zu format is invalid start checking from column %zu",
            line_number, col+1);
    }
}

void csv_keys2cols(size_t cols[], char *keys[], size_t keys_size)
{
    size_t i;
    int ret;
    for (i = 0; i < keys_size; i++) {
        ret = sscanf(keys[i], "%zu", cols + i);
        if (ret != 1) die("csv_keys2col() Error: '%s' can not be converted to index", keys[i]);
    }
}

#ifdef PARSE_TEST
#include <assert.h>
#include <string.h>
/*
 * compile: clang -Wall -g -DPARSE_TEST -o objs/test_parse src/util.c src/parse.c $(pkg-config --libs-only-l json-c)
 */
size_t parse_keys(char *keys[], char *argv, char key_buffer[512])
{
    size_t keys_length = 0;
    char *keys_buffer, *key;

    keys_buffer = e_strdup(argv);
    key = strtok_r(keys_buffer, ", ", &key_buffer);
    keys[keys_length++] = e_strdup(key);
    while ((key = strtok_r(NULL, ", ", &key_buffer))) {
        if (keys_length + 1 > 32) {
            die("parse_keys() Error: keys_buffer overflow you can put more "
                "than 32 keys using this test program");
        }
        keys[keys_length++] = e_strdup(key);
    }
    free(keys_buffer);
    return keys_length;
}

int main(int argc, char *argv[]) {
    char *filename, *format;
    size_t i, j;

    if (argc < 4 || argc > 5) {
        fprintf(stderr,
                "Usage: parse_test FILENAME IN_KEYS OUT_KEYS [FORMAT]\n"
                "\nKeys format:\n"
                "  IN_KEYS: in_key1, in_key2, ...\n"
                "  OUT_KEYS: out_key1, out_key2, ...\n\n");
        return 1;
    }

    filename = argv[1];
    format = NULL;
    if (argc == 5) {
        format = argv[4];
    }

    Array X, y;
    char *in_cols[32], *out_cols[32], keys_buffer[512];
    size_t n_in_cols, n_out_cols;

    n_in_cols = parse_keys(in_cols, argv[2], keys_buffer);
    n_out_cols = parse_keys(out_cols, argv[3], keys_buffer);

    file_read(filename, &X, &y, in_cols, 2, out_cols, 1, true, format);

    for (i = 0; i < X.shape[0]; i++) {
        for (j = 0; j < X.shape[1]; j++) {
            printf("%*.2e\t", 4, X.data[i * X.shape[1] + j]);
        }

        for (j = 0; j < y.shape[1]; j++) {
            if (j == 0) printf("|\t");
            printf("%5.2e", y.data[i * y.shape[1] + j]);;
            if (j < y.shape[1] - 1) printf("\t");
        }
        printf("\n");

    }

    for (i = 0; i < n_in_cols; i++) free(in_cols[i]);
    for (i = 0; i < n_out_cols; i++) free(out_cols[i]);

    return 0;
}
#endif
