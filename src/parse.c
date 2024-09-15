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
#include <stdbool.h>
#include <string.h>
#include <json-c/json.h>

#include "util.h"
#include "parse.h"

#define MAX_FILE_SIZE 536870912 //1<<29; 0.5 GiB

static void json_read(
        FILE *fp,
        Array *input, Array *out,
        struct Configs cfgs,
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

static void json_write(
        FILE *fp,
        Array input, Array out,
        struct Configs cfgs);

static void csv_write(
        FILE *fp,
        Array input, Array out,
        bool write_input,
        char separator,
        int decimal_precision
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
        struct Configs ml_config,
        bool read_output)
{
    FILE *fp;

    char *file_format = ml_config.file_format;

    if (filepath != NULL && strcmp(filepath, "-")) {
        fp = fopen(filepath, "r");
        file_format = file_format_infer(filepath);
    } else {
        fp = fopen("/dev/stdin", "r");
        if (file_format == NULL) {
            die("file_read() Error: file format must be defined");
        }
    }

    if (file_format == NULL) {
        file_format = file_format_infer(filepath);
    }

    /*
    if (!strcmp(file_format, "csv"))        csv_read(fp, input, out, in_keys, n_in_keys, out_keys, n_out_keys, read_output, false, ',');
    else if (!strcmp(file_format, "tsv"))   csv_read(fp, input, out, in_keys, n_in_keys, out_keys, n_out_keys, read_output, false, '\t');
    */
    if (!strcmp(file_format, "json"))  json_read(fp, input, out, ml_config, read_output);
    else {
        die("file_read() Error: unable to parse %s files", file_format);
    }

    fclose(fp);
}

void file_write(Array input, Array out, struct Configs ml_config)
{
    FILE *fp;

    char *filepath = ml_config.out_filepath;
    char *file_format = ml_config.file_format;

    if (filepath != NULL && strcmp(filepath, "-")) {
        fp = fopen(filepath, "w");
        file_format = file_format_infer(filepath);
    } else {
        fp = fopen("/dev/stdout", "w");
        if (file_format == NULL) {
            die("file_write() Error: file format must be defined");
        }
    }

    if (fp == NULL) die("file_write() Error:");

    if (!strcmp(file_format, "json"))       json_write(fp, input, out, ml_config);
    /*
    else if (!strcmp(file_format, "csv"))   csv_write(fp, input, out, write_input, ',', decimal_precision);
    else if (!strcmp(file_format, "tsv"))   csv_write(fp, input, out, write_input, '\t', decimal_precision);
    */
    else {
        die("file_write() Error: unable to write %s files", file_format);
    }
    fclose(fp);
}

void data_postprocess(
        Array *out,
        double *data, size_t data_shape[2],
        struct Configs cfgs,
        bool is_input)
{
    char **keys = (is_input) ? cfgs.input_keys : cfgs.label_keys;
    size_t n_keys = (is_input) ? cfgs.n_input_keys : cfgs.n_label_keys;

    char **categorical_keys = cfgs.categorical_keys;
    size_t n_categorical_keys = cfgs.n_categorical_keys;

    char ***categorical_values = cfgs.categorical_values;
    size_t *n_categorical_values = cfgs.n_categorical_values;

    size_t i, j, data_j;
    for (data_j = j = 0; j < n_keys; j++) {
        int k;
        switch (out->type[j]) {
        case ARRAY_NUMERICAL:
            for (i = 0; i < data_shape[0]; i++) {
                size_t data_index = i * data_shape[1] + data_j;
                size_t index = i * out->shape[1] + j;
                out->data[index].numeric = data[data_index];
            }
            data_j++;
            break;
        case ARRAY_ONEHOT:
            k = util_get_key_index(keys[j], categorical_keys, n_categorical_keys);
            if (k == -1) {
                die("data_postprocess() Error: field '%s' is not registered as categorical",
                    keys[j]);
            }
            for (i = 0; i < data_shape[0]; i++) {
                size_t index = i * out->shape[1] + j;
                size_t data_index = i * data_shape[1] + data_j;
                int onehot_i = util_argmax(data + data_index, n_categorical_values[k]);
                out->data[index].categorical = e_strdup(categorical_values[k][onehot_i]);
            }
            data_j += n_categorical_values[k];
            break;
        default:
            die("data_postprocess() Error: unexpected type received on '%s' field", keys[j]);
        }
    }
}

double * data_preprocess(
        size_t out_shape[2],
        Array data,
        struct Configs cfgs,
        bool is_input,
        bool only_allocate)
{
    double *out;

    char **keys = (is_input) ? cfgs.input_keys : cfgs.label_keys;
    size_t n_keys = (is_input) ? cfgs.n_input_keys : cfgs.n_label_keys;

    char **categorical_keys = cfgs.categorical_keys;
    size_t n_categorical_keys = cfgs.n_categorical_keys;

    char ***categorical_values = cfgs.categorical_values;
    size_t *n_categorical_values = cfgs.n_categorical_values;

    size_t i, j, out_j;

    out_shape[0] = data.shape[0];
    out_shape[1] = 0;
    for (i = 0; i < n_keys; i++) {
        int n;
        switch (data.type[i]) {
        case ARRAY_NUMERICAL:
            out_shape[1]++;
            break;
        case ARRAY_ONEHOT:
            n = util_get_key_index(keys[i], categorical_keys, n_categorical_keys);
            if (n == -1) die("data_preprocess() Error: field '%s' is not marked as categorical", keys[i]);
            out_shape[1] += n_categorical_values[n];
            break;
        default:
            die("data_preprocess() Error: field '%s' has an unknown type", keys[i]);
            break;
        }
    }

    out = ecalloc(out_shape[0] * out_shape[1], sizeof(double));
    if (only_allocate) return out;

    for (out_j = j = 0; j < data.shape[1]; j++) {
        switch (data.type[j]) {
            int k;
            case ARRAY_NUMERICAL:
                for (i = 0; i < out_shape[0]; i++) {
                    size_t index = i * data.shape[1] + j;
                    size_t out_index = i * out_shape[1] + out_j;
                    out[out_index] = data.data[index].numeric;
                }
                out_j++;
                break;
            case ARRAY_ONEHOT:
                k = util_get_key_index(keys[j], categorical_keys, n_categorical_keys);
                for (i = 0; i < out_shape[0]; i++) {
                    int onehot_i;
                    size_t index = i * data.shape[1] + j;
                    onehot_i = util_get_key_index(data.data[index].categorical,
                                                  categorical_values[k],
                                                  n_categorical_values[k]);
                    if (onehot_i == -1) {
                        die("data_preprocess() Error: unexpected '%s' value found",
                            data.data[index].categorical);
                    }
                    size_t out_index = i * out_shape[1] + out_j + onehot_i;
                    out[out_index] = 1.0;
                }
                out_j += n_categorical_values[k];
                break;
            default:
                die("data_preprocess() Error: field '%s' has an unknown type", keys[j]);
        }
    }

    return out;
}

void array_free(Array *x) {
    size_t i, j, index;
    for (j = 1; j < x->shape[1]; j++) {
        switch (x->type[j]) {
        case ARRAY_ORDINAL:
        case ARRAY_ONEHOT:
            for (i = 0; i < x->shape[0]; i++) {
                index = x->shape[1] * i + j;
                free(x->data[index].categorical);
            }
            break;
        default:
            break;
        }
    }
    free(x->type);
    free(x->data);
}

void json_read(
        FILE *fp,
        Array *input, Array *out,
        struct Configs cfgs,
        bool read_output)
{
    static char fp_buffer[MAX_FILE_SIZE];
    size_t i, j, json_obj_length, index;
    json_object *json_obj, *item, *value;
    json_type obj_type;

    char **in_keys = cfgs.input_keys;
    char **out_keys = cfgs.label_keys;
    char **onehot_keys = cfgs.onehot_keys;
    size_t n_input_keys = cfgs.n_input_keys;
    size_t n_out_keys = cfgs.n_label_keys;
    size_t n_onehot_keys = cfgs.n_onehot_keys;


    if (fp == NULL) die("json_read() Error:");

    i = 0;
    do {
        if (i >= MAX_FILE_SIZE) die("json_read() Error: file size is bigger than '%zu'", i, MAX_FILE_SIZE);
        fp_buffer[i] = fgetc(fp);
    } while (fp_buffer[i++] != EOF);

    json_obj = json_tokener_parse(fp_buffer);
    if (!json_object_is_type(json_obj, json_type_array)) {
        die("json_read() Error: unexpected JSON data received, expecting an array");
    }
    json_obj_length = json_object_array_length(json_obj);



    input->shape[0] = (size_t)json_obj_length;
    input->shape[1] = n_input_keys;
    input->type = ecalloc(input->shape[1], sizeof(enum ArrayType));
    input->data = ecalloc(input->shape[0] * input->shape[1], sizeof(input->data[0]));

    out->shape[0] = (size_t)json_obj_length;
    out->shape[1] = n_out_keys;
    out->type = ecalloc(out->shape[1], sizeof(enum ArrayType));
    out->data = ecalloc(out->shape[0] * out->shape[1], sizeof(out->data[0]));

    for (i = 0; i < n_onehot_keys; i++) {
        for (j = 0; j < n_input_keys; j++) {
            if (!strcmp(onehot_keys[i], in_keys[j])) {
                input->type[j] = ARRAY_ONEHOT;
            }
        }

        for (j = 0; j < n_out_keys; j++) {
            if (!strcmp(onehot_keys[i], out_keys[j])) {
                out->type[j] = ARRAY_ONEHOT;
            }
        }
    }


    for (i = 0; i < json_object_array_length(json_obj); i++) {
        item = json_object_array_get_idx(json_obj, i);

        if (!json_object_is_type(item, json_type_object)) {
            die("json_read() Error: unexpected JSON data received, expecting an object");
        }

        if ((size_t)json_object_object_length(item) < n_input_keys + n_out_keys) {
            die("json_read() Error: the number of keys required is greater "
                "than the keys available in the object:\n%s",
                json_object_to_json_string_ext(item, JSON_C_TO_STRING_PRETTY));
        }
        for (j = 0; j < n_input_keys; j++) {
            value = json_object_object_get(item, in_keys[j]);
            obj_type = json_object_get_type(value);
            index = n_input_keys * i + j;
            switch (input->type[j]) {
            case ARRAY_NUMERICAL:
                switch (obj_type) {
                case json_type_int:
                case json_type_double:
                    input->data[index].numeric = json_object_get_double(value);
                    break;
                default:
                    die("json_read() Error: unexpected JSON data received, expecting a number");
                }
                break;
            case ARRAY_ONEHOT:
                switch (obj_type) {
                case json_type_int:
                case json_type_string:
                    input->data[index].categorical = e_strdup(json_object_get_string(value));
                    break;
                default:
                    die("json_read() Error: unexpected JSON data received, expecting a string or integer");
                }
                break;
            default:
                die("json_read() Error: preprocess field type '%s' is not implemented", in_keys[j]);
            }
        }

        if (!read_output) continue;

        for (j = 0; j < n_out_keys; j++) {
            value = json_object_object_get(item, out_keys[j]);
            obj_type = json_object_get_type(value);
            index = n_out_keys * i + j;
            switch (out->type[j]) {
            case ARRAY_NUMERICAL:
                switch (obj_type) {
                case json_type_int:
                case json_type_double:
                    out->data[index].numeric = json_object_get_double(value);
                    break;
                default:
                    die("json_read() Error: unexpected JSON data received, expecting a number");
                }
                break;
            case ARRAY_ONEHOT:
                switch (obj_type) {
                case json_type_int:
                case json_type_string:
                    out->data[index].categorical = e_strdup(json_object_get_string(value));
                    break;
                default:
                    die("json_read() Error: unexpected JSON data received, expecting string or integer");
                }
                break;
            default:
                die("json_read() Error: preprocess field type '%s' is not implemented", out_keys[j]);
            }
        }
    }

    json_object_put(json_obj);
    return;
}


/*
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
*/

void json_write(
        FILE *fp,
        Array input, Array out,
        struct Configs cfgs)
{
    char **in_keys = cfgs.input_keys;
    char **out_keys = cfgs.label_keys;
    size_t n_in_keys = cfgs.n_input_keys;
    size_t n_out_keys = cfgs.n_label_keys;
    bool write_input = !cfgs.only_out;
    int decimal_precision = cfgs.decimal_precision;

    json_object *root = json_object_new_array();
    if (!root) {
        die("json_write() Error: Unable to create json_data");
    }

    if (n_in_keys != input.shape[1])
        die("json_write() Error: input keys and data columns have different sizes");
    if (n_out_keys != out.shape[1])
        die("json_write() Error: output keys and data columns have different sizes");

    size_t i, j;
    for (i = 0; i < input.shape[0]; i++) {
        json_object *obj = json_object_new_object();

        if (write_input) {
            for (j = 0; j < input.shape[1]; j++) {
                char buffer[128];
                size_t index = i * input.shape[1] + j;
                switch (input.type[j]) {
                case ARRAY_NUMERICAL:
                    sprintf(buffer, "%g", input.data[index].numeric);
                    json_object_object_add(obj, in_keys[j], json_object_new_double_s(input.data[index].numeric, buffer));
                    break;
                case ARRAY_ONEHOT:
                    json_object_object_add(obj, in_keys[j], json_object_new_string(input.data[index].categorical));
                    break;
                default:
                    die("json_write(): Unexpected value received");
                }
            }
        }


        for (j = 0; j < out.shape[1]; j++) {
            size_t index = i * out.shape[1] + j;
            char buffer[32];
            switch (out.type[j]) {
            case ARRAY_NUMERICAL:
                sprintf(buffer, "%.*g", decimal_precision, out.data[index].numeric);
                json_object_object_add(obj, out_keys[j], json_object_new_double_s(out.data[index].numeric, buffer));
                break;
            case ARRAY_ONEHOT:
                json_object_object_add(obj, out_keys[j], json_object_new_string(out.data[index].categorical));
                break;
            default:
                die("json_write(): Unexpected value received");
            }
        }
        json_object_array_add(root, obj);
    }
    int ret = fprintf(fp, "%s", json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_SPACED));
    if (ret == -1) {
        die("json_write() Error: unable to write json data");
    }
    json_object_put(root);
}

/*
void csv_write(
        FILE *fp,
        Array input, Array out,
        bool write_input,
        char separator,
        int decimal_precision)
{
    size_t line, col, index;
    for (line = 0; line < input.shape[0]; line++) {
        if (write_input) {
            for (col = 0; col < input.shape[1]; col++) {
                index = input.shape[1] * line + col;
                fprintf(fp, "%g%c", input.data[index], separator);
            }
        }
        for (col = 0; col < out.shape[1]; col++) {
            index = out.shape[1] * line + col;
            fprintf(fp, "%.*g", decimal_precision, out.data[index]);
            if (col == out.shape[1] - 1) continue;
            fprintf(fp, "%c", separator);
        }
        fprintf(fp, "\n");
    }
}
*/

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

char * file_format_infer(char *filename)
{
    char *file_format, *ptr;
    size_t string_length, i;

    string_length = strlen(filename);
    ptr = filename + string_length;
    for (i = string_length; i > 0 && *ptr != '.'; ptr--, i--);
    if (*ptr != '.' || i == 0) {
        die("file_format_infer() Error: unable to infer %s format", filename);
    }
    file_format = ptr + 1;
    return file_format;
}

#ifdef PARSE_TEST
#include <assert.h>
#include <string.h>
/*
 * compile: clang -Wall -Wextra -g -DPARSE_TEST -o objs/test_parse src/util.c src/parse.c $(pkg-config --libs-only-l json-c)
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
    char *in_file, *out_file, *format;
    size_t i, j;

    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "Usage: parse_test IN_FILE OUT_FILE IN_KEYS OUT_KEYS [FORMAT]\n"
                "\nKeys format:\n"
                "  IN_KEYS: in_key1, in_key2, ...\n"
                "  OUT_KEYS: out_key1, out_key2, ...\n\n");
        return 1;
    }

    in_file = argv[1];
    out_file = argv[2];
    format = NULL;
    if (argc == 6) {
        format = argv[5];
    }

    Array X, y;
    char *in_cols[32], *out_cols[32], keys_buffer[512];
    size_t n_in_cols, n_out_cols;

    n_in_cols = parse_keys(in_cols, argv[3], keys_buffer);
    n_out_cols = parse_keys(out_cols, argv[4], keys_buffer);

    struct Configs ml_config = {
        .in_filepath = in_file,
        .out_filepath = out_file,
        .input_keys = in_cols,
        .label_keys = out_cols,
        .n_input_keys = n_in_cols,
        .n_label_keys = n_out_cols,
        .file_format = format,
        .only_out = false,
        .decimal_precision = -1
    };

    file_read(in_file, &X, &y, ml_config, true);

    for (i = 0; i < X.shape[0]; i++) {
        for (j = 0; j < X.shape[1]; j++) {
            fprintf(stderr, "%.2e\t", X.data[i * X.shape[1] + j]);
        }

        for (j = 0; j < y.shape[1]; j++) {
            if (j == 0) fprintf(stderr, "|\t");
            fprintf(stderr, "%.2e", y.data[i * y.shape[1] + j]);;
            if (j < y.shape[1] - 1) printf("\t");
        }
        fprintf(stderr, "\n");

    }

    // use input format if format variable is not defined
    format = (!format && !strcmp(out_file, "-")) ? file_format_infer(in_file) : format;
    file_write(X, y, ml_config);
    for (i = 0; i < n_in_cols; i++) free(in_cols[i]);
    for (i = 0; i < n_out_cols; i++) free(out_cols[i]);

    return 0;
}

#endif
