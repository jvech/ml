#ifndef UTIL_
#define UTIL_

#include <stddef.h>

struct Configs {
    size_t epochs;
    double alpha;
    char **input_keys, **label_keys;
    size_t n_input_keys, n_label_keys;
    char *loss;
    char *in_filepath;
    char *out_filepath;
    char *weights_filepath;
    char *config_filepath;
};

void die(const char *fmt, ...);
void *ecalloc(size_t nmemb, size_t size);
void util_load_cli(struct Configs *ml, int argc, char *argv[]);
void util_load_config(struct Configs *ml, char *filepath);
void util_free_config(struct Configs *ml);
#endif
