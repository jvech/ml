#ifndef __JSON
#define __JSON
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

typedef struct {
    float price;
    float area;
    float latitude;
    float longitude;
} HouseItem;

void json_read(const char *filepath, HouseItem *out);
#endif
