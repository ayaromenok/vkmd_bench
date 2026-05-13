#ifndef ARGS_H
#define ARGS_H

#include <stdint.h>

typedef struct {
    uint32_t matrix_size;
    uint32_t iterations;
    uint32_t device_index;
    int list_devices;
} AppArgs;

AppArgs parse_args(int argc, char** argv);

#endif
