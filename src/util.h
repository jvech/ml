#ifndef UTIL_
#define UTIL_

#include <stdbool.h>
#include <stddef.h>

struct Configs {
    /* net cfgs */
    size_t epochs;
    size_t batch_size;
    double alpha;
    char *loss;
    char **input_keys, **label_keys;
    size_t n_input_keys, n_label_keys;
    char **categorical_keys, ***categorical_values;
    size_t n_categorical_keys, *n_categorical_values;
    char *weights_filepath;
    char *config_filepath;
    bool shuffle;
    /* preprocessing */
    char **onehot_keys;
    size_t n_onehot_keys;
    /* cli cfgs */
    char *file_format;
    char *in_filepath;
    char *out_filepath;
    int decimal_precision;
    bool only_out;
    /* layer cfgs */
    size_t network_size;
    size_t *neurons;
    char **activations;
};

void usage(int exit_code);
void die(const char *fmt, ...);
void *ecalloc(size_t nmemb, size_t size);
void *erealloc(void *ptr, size_t size);
char *e_strdup(const char *s);
int util_get_key_index(char *key, char **keys, size_t n_keys);
int util_argmax(double *values, size_t n_values);
void util_load_cli(struct Configs *ml, int argc, char *argv[]);
void util_load_config(struct Configs *ml, char *filepath);
void util_free_config(struct Configs *ml);
#endif
