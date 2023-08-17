/* See LICENSE file for copyright and license details. */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "util.h"

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

void * ecalloc(size_t nmemb, size_t size)
{
	void *p;

	if (!(p = calloc(nmemb, size)))
		die("calloc:");
	return p;
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
}
