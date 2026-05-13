#include "args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

AppArgs parse_args(int argc, char** argv) {
    AppArgs args = { .matrix_size = 1024, .iterations = 10, .device_index = 0, .list_devices = 0, .save_csv = 0 }; // Default
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ms") == 0 || strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) {
                args.matrix_size = atoi(argv[++i]);
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
        } else if (strcmp(argv[i], "-dl") == 0 || strcmp(argv[i], "--device-list") == 0) {
            args.list_devices = 1;
        } else if (strcmp(argv[i], "-csv") == 0 || strcmp(argv[i], "--save-csv") == 0) {
            args.save_csv = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -ms, --matrix-size <size>   Set matrix size (default: 1024)\n");
            printf("  -i, --iterations <count>    Set benchmarking iterations (default: 10)\n");
            printf("  -d, --device <index>        Select Vulkan device index (default: 0)\n");
            printf("  -dl, --device-list          List available Vulkan devices and exit\n");
            printf("  -csv, --save-csv            Save results to CSV file\n");
            printf("  -h, --help                  Show this help message\n");
            exit(0);
        }
    }
    return args;
}
