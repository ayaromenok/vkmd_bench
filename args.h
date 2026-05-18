#ifndef ARGS_H
#define ARGS_H

#include <stdint.h>

typedef enum {
    DT_FP16,
    DT_INT16,
    DT_FP32,
    DT_INT32
} DataType;

typedef enum {
    OP_MUL,
    OP_ADD,
    OP_SUB,
    OP_DIV,
    OP_MAD
} OperatorType;

typedef struct {
    uint32_t matrix_size;
    uint32_t matrix_start_size;
    uint32_t matrix_step_size;
    uint32_t iterations;
    uint32_t device_index;
    int list_devices;
    int save_csv;
    int multi_bench;
    const char* lact_profile;
    
    // Multi-device select fields
    uint32_t multi_devices[32];
    uint32_t multi_device_count;
    const char* multi_profiles[32];
    uint32_t multi_profile_count;
    
    DataType data_type;
    OperatorType operator_type;
} AppArgs;

AppArgs parse_args(int argc, char** argv);

#endif
