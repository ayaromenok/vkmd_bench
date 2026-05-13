#include "args.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

AppArgs parse_args(int argc, char** argv) {
    AppArgs args = { .matrix_size = 1024 }; // Default
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ms") == 0 || strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) {
                args.matrix_size = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-ms | --matrix-size <size>]\n", argv[0]);
            exit(0);
        }
    }
    return args;
}
