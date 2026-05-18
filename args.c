#include "args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* trim(char* str) {
    while (*str == ' ' || *str == '\t' || *str == '\r' || *str == '\n') {
        str++;
    }
    if (*str == 0) return str;
    char* end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) {
        *end = 0;
        end--;
    }
    return str;
}

static void parse_devices(AppArgs* args, const char* str) {
    args->multi_device_count = 0;
    char* val_copy = strdup(str);
    char* token = strtok(val_copy, ",");
    while (token && args->multi_device_count < 32) {
        char* trimmed = trim(token);
        args->multi_devices[args->multi_device_count++] = atoi(trimmed);
        token = strtok(NULL, ",");
    }
    free(val_copy);
    if (args->multi_device_count > 0) {
        args->device_index = args->multi_devices[0];
    }
}

static void parse_profiles(AppArgs* args, const char* str) {
    args->multi_profile_count = 0;
    char* val_copy = strdup(str);
    char* token = strtok(val_copy, ",");
    while (token && args->multi_profile_count < 32) {
        char* p = token;
        while (*p == ' ' || *p == '\t' || *p == '"' || *p == '\'') p++;
        size_t len = strlen(p);
        while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t' || p[len - 1] == '"' || p[len - 1] == '\'')) {
            p[len - 1] = '\0';
            len--;
        }
        args->multi_profiles[args->multi_profile_count++] = strdup(p);
        token = strtok(NULL, ",");
    }
    free(val_copy);
    if (args->multi_profile_count > 0) {
        args->lact_profile = args->multi_profiles[0];
    }
}

static void parse_operators(AppArgs* args, const char* str) {
    args->multi_operator_count = 0;
    char* val_copy = strdup(str);
    char* token = strtok(val_copy, ",");
    while (token && args->multi_operator_count < 32) {
        char* trimmed = trim(token);
        OperatorType op = OP_MUL;
        if (strcmp(trimmed, "mul") == 0) op = OP_MUL;
        else if (strcmp(trimmed, "add") == 0) op = OP_ADD;
        else if (strcmp(trimmed, "sub") == 0) op = OP_SUB;
        else if (strcmp(trimmed, "div") == 0) op = OP_DIV;
        else if (strcmp(trimmed, "mad") == 0) op = OP_MAD;
        else {
            fprintf(stderr, "Error: Unknown operator '%s' (use mul, add, sub, div, or mad)\n", trimmed);
            exit(1);
        }
        args->multi_operators[args->multi_operator_count++] = op;
        token = strtok(NULL, ",");
    }
    free(val_copy);
    if (args->multi_operator_count > 0) {
        args->operator_type = args->multi_operators[0];
    }
}

static void parse_data_types(AppArgs* args, const char* str) {
    args->multi_data_type_count = 0;
    char* val_copy = strdup(str);
    char* token = strtok(val_copy, ",");
    while (token && args->multi_data_type_count < 32) {
        char* trimmed = trim(token);
        DataType dt = DT_FP16;
        if (strcmp(trimmed, "fp16") == 0) dt = DT_FP16;
        else if (strcmp(trimmed, "int16") == 0) dt = DT_INT16;
        else if (strcmp(trimmed, "fp32") == 0) dt = DT_FP32;
        else if (strcmp(trimmed, "int32") == 0) dt = DT_INT32;
        else {
            fprintf(stderr, "Error: Unknown data type %s (use fp16, int16, fp32, or int32)\n", trimmed);
            exit(1);
        }
        args->multi_data_types[args->multi_data_type_count++] = dt;
        token = strtok(NULL, ",");
    }
    free(val_copy);
    if (args->multi_data_type_count > 0) {
        args->data_type = args->multi_data_types[0];
    }
}

static void load_from_ini(AppArgs* args, const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        return;
    }

    printf("Loading settings from %s...\n", filepath);
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);
        if (trimmed[0] == '\0' || trimmed[0] == ';' || trimmed[0] == '#' || trimmed[0] == '[') {
            continue;
        }

        char* eq = strchr(trimmed, '=');
        if (!eq) continue;

        *eq = '\0';
        char* key = trim(trimmed);
        char* val = trim(eq + 1);

        if (strcmp(key, "matrix-size") == 0 || strcmp(key, "matrix_size") == 0 || strcmp(key, "ms") == 0) {
            args->matrix_size = atoi(val);
        } else if (strcmp(key, "matrix-start-size") == 0 || strcmp(key, "matrix_start_size") == 0 || strcmp(key, "mss") == 0) {
            args->matrix_start_size = atoi(val);
        } else if (strcmp(key, "matrix-increment-step") == 0 || strcmp(key, "matrix_increment_step") == 0 || strcmp(key, "mis") == 0) {
            args->matrix_step_size = atoi(val);
        } else if (strcmp(key, "iterations") == 0 || strcmp(key, "i") == 0) {
            args->iterations = atoi(val);
        } else if (strcmp(key, "device") == 0 || strcmp(key, "d") == 0) {
            parse_devices(args, val);
        } else if (strcmp(key, "data-type") == 0 || strcmp(key, "data_type") == 0 || strcmp(key, "dt") == 0) {
            parse_data_types(args, val);
        } else if (strcmp(key, "device-list") == 0 || strcmp(key, "device_list") == 0 || strcmp(key, "dl") == 0) {
            args->list_devices = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0 || strcmp(val, "yes") == 0);
        } else if (strcmp(key, "operator") == 0 || strcmp(key, "o") == 0) {
            parse_operators(args, val);
        } else if (strcmp(key, "save-csv") == 0 || strcmp(key, "save_csv") == 0 || strcmp(key, "csv") == 0) {
            args->save_csv = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0 || strcmp(val, "yes") == 0);
        } else if (strcmp(key, "multi-bench") == 0 || strcmp(key, "multi_bench") == 0 || strcmp(key, "mb") == 0) {
            args->multi_bench = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0 || strcmp(val, "yes") == 0);
        } else if (strcmp(key, "lact") == 0 || strcmp(key, "l") == 0) {
            parse_profiles(args, val);
        }
    }
    fclose(file);
}

AppArgs parse_args(int argc, char** argv) {
    AppArgs args = { 
        .matrix_size = 1024, 
        .matrix_start_size = 32,
        .matrix_step_size = 32,
        .iterations = 10, 
        .device_index = 0, 
        .list_devices = 0, 
        .save_csv = 0,
        .multi_bench = 0,
        .lact_profile = "210_405",
        .data_type = DT_FP16,
        .operator_type = OP_MUL,
        .multi_operator_count = 0,
        .multi_data_type_count = 0
    };
    
    // Load options from settings.ini if present, which can then be overridden by cmd line args
    load_from_ini(&args, "settings.ini");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ms") == 0 || strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) {
                args.matrix_size = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-mss") == 0 || strcmp(argv[i], "--matrix-start-size") == 0) {
            if (i + 1 < argc) {
                args.matrix_start_size = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-mis") == 0 || strcmp(argv[i], "--matrix-increment-step") == 0) {
            if (i + 1 < argc) {
                args.matrix_step_size = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) {
                args.iterations = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                parse_devices(&args, argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--data-type") == 0) {
            if (i + 1 < argc) {
                parse_data_types(&args, argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-dl") == 0 || strcmp(argv[i], "--device-list") == 0) {
            args.list_devices = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--operator") == 0) {
            if (i + 1 < argc) {
                parse_operators(&args, argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-csv") == 0 || strcmp(argv[i], "--save-csv") == 0) {
            args.save_csv = 1;
        } else if (strcmp(argv[i], "-mdb") == 0 || strcmp(argv[i], "--multi-device-bench") == 0) {
            args.multi_bench = 1;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lact") == 0) {
            if (i + 1 < argc) {
                parse_profiles(&args, argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -ms, --matrix-size <size>              Set max matrix size (default: 1024)\n");
            printf("  -mss, --matrix-start-size <sz>         Set start matrix size (default: 32)\n");
            printf("  -mis, --matrix-increment-step <step>   Set matrix incrementstep (default: 32)\n");
            printf("  -i, --iterations <count>               Set benchmarking iterations (default: 10)\n");
            printf("  -d, --device <index>                   Select Vulkan device index (default: 0)\n");
            printf("  -dt, --data-type <type>                Select data type: fp16, int16, fp32, int32 (default: fp16)\n");
            printf("  -dl, --device-list                     List available Vulkan devices and exit\n");
            printf("  -o, --operator <op>                    Select operator: mul, add, sub, div, mad (default: mul)\n");
            printf("  -csv, --save-csv                       Save results to CSV file\n");
            printf("  -mdb, --multi-device-bench             Benchmark device 0 and 2 side-by-side\n");
            printf("  -l, --lact <profile>                   Set LACT profile name string (default: 210_405)\n");
            printf("  -h, --help                             Show this help message\n");
            exit(0);
        } else {
            fprintf(stderr, "Warning: Unrecognized command-line parameter '%s'\n", argv[i]);
        }
    }

    if (args.multi_bench && args.multi_device_count == 0) {
        args.multi_device_count = 2;
        args.multi_devices[0] = 0;
        args.multi_devices[1] = 2;
    }
    if (args.multi_bench && args.multi_profile_count == 0) {
        args.multi_profile_count = 2;
        args.multi_profiles[0] = strdup("0_210_405");
        args.multi_profiles[1] = strdup("2_210_405");
    }

    if (args.multi_operator_count == 0) {
        args.multi_operator_count = 1;
        args.multi_operators[0] = args.operator_type;
    }

    if (args.multi_data_type_count == 0) {
        args.multi_data_type_count = 1;
        args.multi_data_types[0] = args.data_type;
    }

    return args;
}
