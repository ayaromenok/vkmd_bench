### Intro

Small Vulkan benchmark

### Benchmark
 - Vulkan
 - MUL, ADD, SUB, DIV, MAD
 - fp16/32, int16/32

### Dual bench

```
--- Starting Dual Device Benchmarking ---
Configuring Device 0 (RTX A4000) profile: "0_210_405"
0_210_405
Running benchmarks on Device 0...

Configuring Device 2 (CMP 70HX) profile: "2_210_405"
2_210_405
Running benchmarks on Device 2...
```

#### Dual Benchmark Results: MUL FP16

| Matrix Size | Device 0 (RTX A4000) [GFLOPS] | Device 2 (CMP 70HX) [GFLOPS] |
| :--- | :---: | :---: |
| 256 x 256 | 91.148 | 24.054 |
| 512 x 512 | 130.216 | 67.298 |
| 768 x 768 | 142.123 | 84.150 |
| 1024 x 1024 | 138.231 | 88.284 |


### Options
  -ms, --matrix-size <size>              Set max matrix size (default: 1024)
  -mss, --matrix-start-size <sz>         Set start matrix size (default: 32)
  -mis, --matrix-increment-step <step>   Set matrix incrementstep (default: 32)
  -i, --iterations <count>               Set benchmarking iterations (default: 10)
  -d, --device <index>                   Select Vulkan device index (default: 0)
  -dt, --data-type <type>                Select data type: fp16, int16, fp32, int32 (default: fp16)
  -dl, --device-list                     List available Vulkan devices and exit
  -o, --operator <op>                    Select operator: mul, add, sub, div, mad (default: mul)
  -csv, --save-csv                       Save results to CSV file
  -db, --dual-bench                      Benchmark device 0 and 2 side-by-side
  -h, --help                             Show this help message
 
