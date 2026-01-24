#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tensor/tensor.h"

static int count_columns(const char* line) {
    int count = 0;
    const char* tmp = line;
    while (*tmp) {
        if (*tmp == ',') count++;
        tmp++;
    }
    return count + 1;
}
static int count_rows(FILE* fp) {
    int rows = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) rows++;
    fseek(fp, 0, SEEK_SET);
    return rows;
}

Tensor* tensor_from_csv(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) return NULL;

    char buffer[1024];
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return NULL;
    }

    int cols = count_columns(buffer);
    int rows = count_rows(fp);
    rewind(fp);

    float* data = malloc(sizeof(float) * rows * cols);
    if (!data) {
        fclose(fp);
        return NULL;
    }

    int r = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        char* token = strtok(buffer, ",\n");
        int c = 0;
        while (token) {
            data[r * cols + c] = strtof(token, NULL);
            token = strtok(NULL, ",\n");
            c++;
        }
        r++;
    }

    fclose(fp);
    Tensor* t = malloc(sizeof(Tensor));
    t->ndim = 2;
    t->shape = malloc(sizeof(int) * 2);
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->data = data;
    t->grad = NULL;

    return t;
}
