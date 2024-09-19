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
#include <errno.h>

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
        struct Configs cfgs,
        bool read_output,
        char *separator
        );

static void json_write(
        FILE *fp,
        Array input, Array out,
        struct Configs cfgs
        );

static void csv_write(
        FILE *fp,
        Array input, Array out,
        struct Configs cfgs,
        char *separator
        );

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

    if (!strcmp(file_format, "csv"))        csv_read(fp, input, out, ml_config, read_output, ",");
    else if (!strcmp(file_format, "tsv"))   csv_read(fp, input, out, ml_config, read_output, "\t");
    else if (!strcmp(file_format, "json"))  json_read(fp, input, out, ml_config, read_output);
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
    else if (!strcmp(file_format, "csv"))   csv_write(fp, input, out, ml_config, ",");
    else if (!strcmp(file_format, "tsv"))   csv_write(fp, input, out, ml_config, "\t");
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


void csv_read(
        FILE *fp,
        Array *input,
        Array *out,
        struct Configs cfgs,
        bool read_output,
        char *separator)
{
    char *line = NULL, *line_buffer, **values_buffer;
    size_t line_number = 0, line_size = 0;
    size_t n_values_buffer;
    size_t *in_indexes, *out_indexes;
    bool has_header = true;

    char **in_keys, **out_keys, **onehot_keys;
    size_t n_in_keys, n_out_keys, n_onehot_keys;

    in_keys = cfgs.input_keys;
    out_keys = cfgs.label_keys;
    onehot_keys = cfgs.onehot_keys;

    n_in_keys = cfgs.n_input_keys;
    n_out_keys = cfgs.n_label_keys;
    n_onehot_keys = cfgs.n_onehot_keys;

    n_values_buffer = n_in_keys + n_out_keys;
    values_buffer = ecalloc(n_values_buffer, sizeof(char *));

    in_indexes = ecalloc(n_in_keys, sizeof(size_t));
    out_indexes = ecalloc(n_out_keys, sizeof(size_t));

    if (fp == NULL) die("csv_read() Error:");

    input->type = ecalloc(n_in_keys, sizeof(enum ArrayType));
    out->type = ecalloc(n_out_keys, sizeof(enum ArrayType));
    input->data = NULL;
    out->data = NULL;

    input->shape[0] = out->shape[0] = 0;
    input->shape[1] = n_in_keys;
    out->shape[1] = n_out_keys;

    for (size_t i = 0; i < n_in_keys; i++) {
        int ret = util_get_key_index(in_keys[i], onehot_keys, n_onehot_keys);
        if (ret >= 0) input->type[i] = ARRAY_ONEHOT;
    }

    for (size_t i = 0; i < n_out_keys; i++) {
        int ret = util_get_key_index(out_keys[i], onehot_keys, n_onehot_keys);
        if (ret >= 0) out->type[i] = ARRAY_ONEHOT;
    }

    while (getline(&line, &line_size, fp) != -1) {
        /* Get line values */
        char *value;
        size_t cols = 0;
        line_buffer = line;
        *(strstr(line, "\n")) = '\0'; //strip new line character e.g ("line text\n" -> "line text")
        while ((value = strsep(&line_buffer, separator))) {
            if (cols == n_values_buffer && line_number == 0) {
                n_values_buffer++;
                values_buffer = erealloc(values_buffer, n_values_buffer * sizeof(char *));
            } else if (cols == n_values_buffer) {
                die("csv_read() Error: line %d has different columns than other lines", line_number);
            }
            values_buffer[cols++] = value;
        }

        /* Set up keys indexes */
        if (line_number == 0) {
            size_t i;
            int key_index;

            for (i = 0; i < n_in_keys && has_header; i++) {
                key_index = util_get_key_index(in_keys[i], values_buffer, n_values_buffer);
                if (key_index == -1) has_header = false;
            }

            for (i = 0; i < n_out_keys && read_output && has_header; i++) {
                key_index = util_get_key_index(out_keys[i], values_buffer, n_values_buffer);
                if (key_index == -1) has_header = false;
            }

            for (i = 0; i < n_in_keys; i++) {
                key_index = util_get_key_index(in_keys[i], values_buffer, n_values_buffer);
                in_indexes[i] = has_header ? (size_t)key_index : i;
            }

            for (i = 0; i < n_out_keys && read_output; i++) {
                key_index = util_get_key_index(out_keys[i], values_buffer, n_values_buffer);
                out_indexes[i] = has_header ? (size_t)key_index : i + n_in_keys;
            }
        }

        if (has_header && !line_number) {
            line_number++;
            continue;
        }

        /* Allocate memory for the data */
        input->data = erealloc(input->data, (input->shape[0] + 1) * n_in_keys * sizeof(union ArrayValue));
        out->data = erealloc(out->data, (out->shape[0] + 1) * n_out_keys * sizeof(union ArrayValue));

        /* Fill the data */
        int ret;
        size_t i, j, index;
        for (i = 0; i < n_in_keys; i++) {
            ret = 0;
            j = in_indexes[i];
            index = input->shape[0] * n_in_keys + i;
            switch (input->type[i]) {
            case ARRAY_NUMERICAL:
                ret = sscanf(values_buffer[j], "%lf", &input->data[index].numeric);
                if (ret < 1) die("csv_read() Error: expecting a number not '%s'", values_buffer[j]);
                break;
            case ARRAY_ONEHOT:
                ret = sscanf(values_buffer[j], "%lf", &input->data[index].numeric);
                if (ret >= 1) die("csv_read() Error: expecting a string or integer not '%s'", values_buffer[j]);
                input->data[index].categorical = e_strdup(values_buffer[j]);
                break;
            default:
                die("csv_read() Error: field '%s' has an unexpected type", in_keys[i]);
            }
        }

        for (i = 0; i < n_out_keys && read_output; i++) {
            ret = 0;
            j = out_indexes[i];
            index = out->shape[0] * n_out_keys + i;
            switch (out->type[i]) {
            case ARRAY_NUMERICAL:
                ret = sscanf(values_buffer[j], "%lf", &out->data[index].numeric);
                if (ret < 1) die("csv_read() Error: expecting a number not '%s'", values_buffer[j]);
                break;
            case ARRAY_ONEHOT:
                out->data[index].categorical = e_strdup(values_buffer[j]);
                break;
            default:
                die("csv_read() Error: field '%s' has an unexpected type", out_keys[i]);
            }
        }
        input->shape[0]++;
        out->shape[0]++;
        line_number++;
    }

    if (errno != 0) die("csv_read() Error:");

    free(line);
    free(in_indexes);
    free(out_indexes);
    free(values_buffer);
}

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

void csv_write(
        FILE *fp,
        Array input, Array out,
        struct Configs cfgs,
        char *separator)
{
    int decimal_precision = cfgs.decimal_precision;
    bool write_input = !cfgs.only_out;

    size_t i,j,index;

    for (j = 0; j < cfgs.n_input_keys && write_input; j++) {
        fprintf(fp, "%s%s", cfgs.input_keys[j], separator);
    }

    for (j = 0; j < cfgs.n_label_keys; j++) {
        fprintf(fp, "%s", cfgs.label_keys[j]);

        if (j == cfgs.n_label_keys - 1) fprintf(fp, "\n");
        else fprintf(fp, "%s", separator);
    }

    for (i = 0; i < input.shape[0]; i++) {
        for (j = 0; j < input.shape[1] && write_input; j++) {
            index = i * out.shape[1] + j;
            switch (input.type[j] ) {
            case ARRAY_NUMERICAL:
                fprintf(fp, "%.*g%s", decimal_precision, input.data[index].numeric, separator);
                break;
            case ARRAY_ONEHOT:
                fprintf(fp, "%s%s", input.data[index].categorical, separator);
                break;
            default:
                die("csv_write() Error: Unexpected type found on field '%s'", cfgs.input_keys[j]);
            }
        }

        for (j = 0; j < out.shape[1]; j++) {
            index = i * out.shape[1] + j;
            switch (out.type[j] ) {
            case ARRAY_NUMERICAL:
                fprintf(fp, "%.*g", decimal_precision, out.data[index].numeric);
                break;
            case ARRAY_ONEHOT:
                fprintf(fp, "%s", out.data[index].categorical);
                break;
            default:
                die("csv_write() Error: Unexpected type found on field '%s'", cfgs.label_keys[j]);
            }
            if (j == out.shape[1] - 1) fprintf(fp, "\n");
            else fprintf(fp, "%s", separator);
        }
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
