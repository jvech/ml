#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <json-c/json.h>

#include "nn.h"

const size_t MAX_FILE_SIZE = 1<<29; // 0.5 GiB

typedef struct Array {
    double *data;
    size_t shape[2];
} Array;

#define ARRAY_SIZE(x, type) sizeof(x) / sizeof(type)

static void version()
{
    printf("ml 0.1\n");
    printf("Written by vech\n");
    exit(0);
}

static void usage(int exit_code)
{
    FILE *fp = (!exit_code) ? stdout : stderr;
    fprintf(fp,
            "Usage: ml train [Options] {[-i INPUT]}... {[-l LABEL]}... JSON_FILE\n"
            "   or: ml predict [-o FILE] FILE\n"
            "Train and predict json data\n"
            "\n"
            "Options:\n"
            "  -a, --alpha=ALPHA        Learning rate (only works with train) [default: 1e-5]\n"
            "  -e, --epochs=EPOCHS      Number of epochs to train the model (only works with train)\n"
            "                           [default: 100]\n"
            "  -i INPUT                 Input key from json file (only works with train)\n"
            "  -l LABEL                 Label key from json file (only works with train)\n"
            "  -o, --output FILE        Output file (only works with predict)\n"
            "\n"
            "Examples:\n"
            "  $ ml train -i area -i latitude -i longitude -l price housing.json\n"
            "  $ ml predict housing.json -o predictions.json\n"
           );
    exit(exit_code);
}

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

int main(int argc, char *argv[]) {
    char **input_keys, **label_keys;
    char *out_filename, *in_filename;
    double epochs = 100, alpha = 1e-5;

    static struct option long_opts[] = {
        {"help",        no_argument,        0, 'h'},
        {"version",     no_argument,        0, 'v'},
        {"epochs",      required_argument,  0, 'e'},
        {"alpha",       required_argument,  0, 'a'},
        {"output",      required_argument,  0, 'o'},
        {0,             0,                  0,  0 },
    };
    int c;

    while (1) {
        c = getopt_long(argc, argv, "hve:a:o:i:l:", long_opts, NULL);
        printf("optind: %d\n", optind);
        if (c == -1) {
            break;
        }
        switch (c) {
        case 'h':
            usage(0);
        case 'v':
            version();
        case 'e':
            printf("epochs: %s\n", optarg);
            break;
        case 'a':
            printf("alpha: %s\n", optarg);
            break;
        case 'o':
            printf("output: '%s'\n", optarg);
            break;
        case 'i':
            printf("input: '%s'\n", optarg);
            break;
        case 'l':
            printf("label: '%s'\n", optarg);
            break;
        default:
            usage(1);
        }
    }

    return 0;
}
