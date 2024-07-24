#ifndef UTIL_
#define UTIL_

#include <stddef.h>

struct Configs {
    /* net cfgs */
    size_t epochs;
    double alpha;
    char *loss;
    char **input_keys, **label_keys;
    size_t n_input_keys, n_label_keys;
    char *weights_filepath;
    char *config_filepath;
    /* cli cfgs */
    char *in_filepath;
    char *out_filepath;
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
void util_load_cli(struct Configs *ml, int argc, char *argv[]);
void util_load_config(struct Configs *ml, char *filepath);
void util_free_config(struct Configs *ml);
#endif
