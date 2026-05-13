#ifndef ARGS_H
#define ARGS_H

#include <stdint.h>

typedef enum {
    DT_FP16,
    DT_INT16,
    DT_FP32,
    DT_INT32
} DataType;

typedef struct {
    uint32_t matrix_size;
    uint32_t iterations;
    uint32_t device_index;
    int list_devices;
    int save_csv;
    DataType data_type;
} AppArgs;

AppArgs parse_args(int argc, char** argv);

#endif
