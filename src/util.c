/* See LICENSE file for copyright and license details. */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "util.h"
#define BUFFER_SIZE 1024

static char ** config_read_values(size_t *n_out_keys, char *first_value, char **strtok_ptr);
static void load_net_cfgs(struct Configs *cfg, char *key, char *value, char *strtok_ptr, char *filepath);
static void load_lyr_cfgs(struct Configs *cfg, char *key, char *value, char *filepath);
static void add_lyr(struct Configs *cfg);

void die(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	if (fmt[0] && fmt[strlen(fmt)-1] == ':') {
		fputc(' ', stderr);
		perror(NULL);
	} else {
		fputc('\n', stderr);
	}

	exit(1);
}

void * erealloc(void *ptr, size_t size)
{
	void *p;

	if (!(p = realloc(ptr, size)))
		die("realloc:");
	return p;
}

void * ecalloc(size_t nmemb, size_t size)
{
	void *p;

	if (!(p = calloc(nmemb, size)))
		die("calloc:");
	return p;
}

char *e_strdup(const char *s)
{
    char *out = strdup(s);
    if (out == NULL) die("strdup() Error:");
    return out;
}


void version()
{
    printf("ml 0.1\n");
    printf("Written by vech\n");
    exit(0);
}

void usage(int exit_code)
{
    FILE *fp = (!exit_code) ? stdout : stderr;
    fprintf(fp,
            "Usage: ml train [Options] JSON_FILE\n"
            "   or: ml predict [-o FILE] FILE\n"
            "Train and predict json data\n"
            "\n"
            "Options:\n"
            "  -a, --alpha=ALPHA        Learning rate (only works with train) [default: 1e-5]\n"
            "  -e, --epochs=EPOCHS      Number of epochs to train the model (only works with train)\n"
            "                           [default: 100]\n"
            "  -o, --output FILE        Output file (only works with predict)\n"
            "\n"
            "Examples:\n"
            "  $ ml train -e 150 -a 1e-4 housing.json\n"
            "  $ ml predict housing.json -o predictions.json\n"
           );
    exit(exit_code);
}

void util_load_cli(struct Configs *ml, int argc, char *argv[])
{
    if (argc <= 1) usage(1);
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

        if (c == -1) {
            break;
        }
        switch (c) {
        case 'e':
            ml->epochs = (size_t)atol(optarg);
            break;
        case 'a':
            ml->alpha = (double)atof(optarg);
            break;
        case 'o':
            ml->out_filepath = optarg;
            break;
        case 'h':
            usage(0);
        case 'v':
            version();
        default:
            usage(1);
        }
    }

    argv += optind;
    argc -= optind;
    if (argc != 2) usage(1);

    ml->in_filepath = argv[1];
}

void util_free_config(struct Configs *ml)
{
    if (ml->loss != NULL) free(ml->loss);
    if (ml->neurons != NULL) free(ml->neurons);
    if (ml->weights_filepath != NULL) free(ml->weights_filepath);

    if (ml->input_keys != NULL) {
        for (size_t i = 0; i < ml->n_input_keys; i++)
            free(ml->input_keys[i]);
        free(ml->input_keys);
    }

    if (ml->label_keys != NULL) {
        for (size_t i = 0; i < ml->n_label_keys; i++)
            free(ml->label_keys[i]);
        free(ml->label_keys);
    }

    if (ml->activations != NULL) {
        for (size_t i = 0; i < ml->network_size; i++)
            free(ml->activations[i]);
        free(ml->activations);
    }
}

void util_load_config(struct Configs *ml, char *filepath)
{
    enum Section {NET, LAYER, OUT_LAYER};
    enum Section section;
    int line_number = 0;
    char line_buffer[BUFFER_SIZE], line_buffer_original[BUFFER_SIZE];
    char token_buffer[BUFFER_SIZE];
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL) return;

    while (fgets(line_buffer, BUFFER_SIZE, fp)) {
        int ret = sscanf(line_buffer, "[%[-_a-zA-Z0-9]]", token_buffer);
        line_number++;
        if (ret >= 1){
            if  (!strcmp("net", token_buffer)) {
                section = NET;
            } else if (!strcmp("layer", token_buffer)) {
                section = LAYER;
                ml->network_size++;
                add_lyr(ml);
            } else if (!strcmp("outlayer", token_buffer)) {
                section = OUT_LAYER;
                ml->network_size++;
                add_lyr(ml);
                ml->neurons[ml->network_size-1] = ml->n_label_keys;
            }
            continue;
        }

        sscanf(line_buffer, "%1023[^\n]", line_buffer_original);


        char *line_ptr = line_buffer;
        while (*line_ptr == ' ') line_ptr++; // delete whitespaces

        /* if the line start with comments or is a blank line ignore it */
        if (*line_ptr == ';'
            || *line_ptr == '#'
            || *line_ptr == '\n') continue;

        /* Verify that each line starts with [a-zA-Z] */
        if ((*line_ptr < 0x41 && *line_ptr > 0x5A)
            || (*line_ptr < 0x61 && *line_ptr > 0x7A))
            goto util_load_config_error;

        /* Load Key Value */
        char *key, *value;
        char *ptr_buffer;
        strtok_r(line_buffer, ";#", &ptr_buffer); // omit comments
        key = strtok_r(line_buffer, " =", &ptr_buffer);
        value = strtok_r(NULL, "= ,\n", &ptr_buffer);
        if (value == NULL) goto util_load_config_error;

        switch (section) {
        case NET:
            load_net_cfgs(ml, key, value, ptr_buffer, filepath);
            break;
        case LAYER:
            load_lyr_cfgs(ml, key, value, filepath);
            break;
        case OUT_LAYER:
            load_lyr_cfgs(ml, key, value, filepath);
            if (!strcmp("neurons", key) && atof(value) != ml->n_label_keys) {
                die("util_load_config() Error: out layer neurons '%zu' differ from the number of labels '%zu'",
                    ml->n_label_keys, atof(value));
            }
            break;
        default:
            goto util_load_config_error;
            break;
        }
    }
    fclose(fp);
    return;

util_load_config_error:
    die("util_load_config() Error: Invalid format on file %s.\n  %d: %s",
        filepath, line_number, line_buffer_original);
}

void add_lyr(struct Configs *cfg)
{
    if (cfg->network_size == 1) {
        cfg->activations = ecalloc(1, sizeof(char *));
        cfg->neurons = ecalloc(1, sizeof(size_t));
        return;
    }
    cfg->activations = erealloc(cfg->activations, cfg->network_size * sizeof(char *));
    cfg->neurons = erealloc(cfg->neurons, cfg->network_size * sizeof(size_t));
}
void load_lyr_cfgs(struct Configs *cfg, char *key, char *value, char *filepath)
{
    size_t index = cfg->network_size - 1;
    if (index > cfg->network_size)
        die("load_lyr_cfgs() Error: index '%d' is greater than network_size '%d'", index, cfg->network_size);

    if (!strcmp(key, "activation"))     cfg->activations[index] = strdup(value);
    else if (!strcmp(key, "neurons"))   cfg->neurons[index] = atof(value);
    else die("util_load_config() Error: Unknown parameter '%s' on file %s.", key, filepath);

}

void load_net_cfgs(struct Configs *cfg, char *key, char *value, char *strtok_ptr, char *filepath)
{
    if (!strcmp(key, "weights_path"))   cfg->weights_filepath = e_strdup(value);
    else if (!strcmp(key, "loss"))      cfg->loss = e_strdup(value);
    else if (!strcmp(key, "epochs"))    cfg->epochs = (size_t)atol(value);
    else if (!strcmp(key, "alpha"))     cfg->alpha = (double)atof(value);
    else if (!strcmp(key, "inputs"))    cfg->input_keys = config_read_values(&(cfg->n_input_keys), value, &strtok_ptr);
    else if (!strcmp(key, "labels"))    cfg->label_keys = config_read_values(&(cfg->n_label_keys), value, &strtok_ptr);
    else die("util_load_config() Error: Unknown parameter '%s' on file %s.", key, filepath);
}

char ** config_read_values(size_t *n_out_keys, char *first_value, char **strtok_ptr)
{
    *n_out_keys = 1;
    char **out_keys = ecalloc(1, sizeof(char *));
    out_keys[0] = e_strdup(first_value);

    char *value;
    while ((value = strtok_r(NULL, ", \n", strtok_ptr)) != NULL) {
        out_keys = erealloc(out_keys, sizeof(char *) * (*n_out_keys + 1));
        out_keys[*n_out_keys] = e_strdup(value);
        (*n_out_keys)++;
    }
    return out_keys;
}
#undef BUFFER_SIZE
