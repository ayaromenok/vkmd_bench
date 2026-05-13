#ifndef ARGS_H
#define ARGS_H

#include <stdint.h>

typedef struct {
    uint32_t matrix_size;
} AppArgs;

AppArgs parse_args(int argc, char** argv);

#endif
