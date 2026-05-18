#include "args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        .operator_type = OP_MUL
    };
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
                args.device_index = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--data-type") == 0) {
            if (i + 1 < argc) {
                char* dt = argv[++i];
                if (strcmp(dt, "fp16") == 0) args.data_type = DT_FP16;
                else if (strcmp(dt, "int16") == 0) args.data_type = DT_INT16;
                else if (strcmp(dt, "fp32") == 0) args.data_type = DT_FP32;
                else if (strcmp(dt, "int32") == 0) args.data_type = DT_INT32;
                else {
                    fprintf(stderr, "Error: Unknown data type %s (use fp16, int16, fp32, or int32)\n", dt);
                    exit(1);
                }
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-dl") == 0 || strcmp(argv[i], "--device-list") == 0) {
            args.list_devices = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--operator") == 0) {
            if (i + 1 < argc) {
                char* op = argv[++i];
                if (strcmp(op, "mul") == 0) args.operator_type = OP_MUL;
                else if (strcmp(op, "add") == 0) args.operator_type = OP_ADD;
                else if (strcmp(op, "sub") == 0) args.operator_type = OP_SUB;
                else if (strcmp(op, "div") == 0) args.operator_type = OP_DIV;
                else if (strcmp(op, "mad") == 0) args.operator_type = OP_MAD;
                else {
                    fprintf(stderr, "Error: Unknown operator '%s' (use mul, add, sub, div, or mad)\n", op);
                    exit(1);
                }
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-csv") == 0 || strcmp(argv[i], "--save-csv") == 0) {
            args.save_csv = 1;
        } else if (strcmp(argv[i], "-mb") == 0 || strcmp(argv[i], "--multi-bench") == 0) {
            args.multi_bench = 1;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lact") == 0) {
            if (i + 1 < argc) {
                args.lact_profile = argv[++i];
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
            printf("  -mb, --multi-bench                     Benchmark device 0 and 2 side-by-side\n");
            printf("  -l, --lact <profile>                   Set LACT profile name string (default: 210_405)\n");
            printf("  -h, --help                             Show this help message\n");
            exit(0);
        } else {
            fprintf(stderr, "Warning: Unrecognized command-line parameter '%s'\n", argv[i]);
        }
    }
    return args;
}
