#include "json.h"

static void fill_item(HouseItem *item, char *key_buffer, char *value_buffer);
static void fill_buffer(FILE *fp, char *key_buffer, char *value_buffer);

void json_read(const char *filepath, HouseItem *out)
{
    FILE *fp;
    fp = fopen(filepath, "r");
    if (fp == NULL) {
        perror("json_read() Error");
        exit(1);
    }

    char c, key_buffer[1024], value_buffer[1024];
    size_t out_size = 0;
    while ((c = fgetc(fp)) != EOF) {
        switch (c) {
        }
    }
}
