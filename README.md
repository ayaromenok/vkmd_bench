# VulKan Multi-Device benchmark

Vulkan benchmark which allow to test devices side-by-side. Originally intended to evaluate NVidia CMP HX series with consumer market devices. 

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

#### Multi Device Results: MUL FP16

| Matrix Size | Device 0 (NVIDIA RTX A4000) [GFLOPS] | Device 2 (NVIDIA CMP 70HX) [GFLOPS] | Device 4 (AMD Radeon Graphics (RADV RENOIR)) [GFLOPS] |
| :--- | :---: | :---: | :---: |
| 256 x 256 | 87.855 | 24.026 | 88.362 |
| 512 x 512 | 132.055 | 63.706 | 101.174 |
| 768 x 768 | 140.372 | 82.315 | 81.207 |
| 1024 x 1024 | 144.080 | 87.604 | 77.426 |

##### Test Hardware

|name|SM| fp16, GFLOPS | fp32,GFLOPS | RAM badwidth|
|---|---|---|---|---|
|HX|3840|1612|1612|~25.8|
|A4|6144|2580|2580|~25.8|
|V7|448| 3404|1702|~56|

|chip|name|short name| fp16, GFLOPS | fp32,GFLOPS 
|---|---|---|---|---|
|[Ampere](https://www.techpowerup.com/gpu-specs/?architecture=Ampere) [GA104](https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964) | [NVIDIA CMP 70HX](https://www.techpowerup.com/gpu-specs/cmp-70hx.c3822)| HX | 10 710 |  10 710 | 
|[Ampere](https://www.techpowerup.com/gpu-specs/?architecture=Ampere) [GA104](https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964) | [NVIDIA Quadro RTX A4000](https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756)| A4 | 19 170 | 19 170 |
|[GCN5](https://www.techpowerup.com/gpu-specs/?architecture=GCN%205.0)|[Cezanne/Vega7](https://www.techpowerup.com/cpu-specs/ryzen-5-5600g.c2471)| V7 | ||
### Options

```
  -ms, --matrix-size <size>              Set max matrix size (default: 1024)
  -mss, --matrix-start-size <sz>         Set start matrix size (default: 32)
  -mis, --matrix-increment-step <step>   Set matrix incrementstep (default: 32)
  -i, --iterations <count>               Set benchmarking iterations (default: 10)
  -d, --device <index>                   Select Vulkan device index (default: 0)
  -dt, --data-type <type>                Select data type: fp16, int16, fp32, int32 (default: fp16)
  -dl, --device-list                     List available Vulkan devices and exit
  -o, --operator <op>                    Select operator: mul, add, sub, div, mad (default: mul)
  -csv, --save-csv                       Save results to CSV file
  -mdb, --multi-device-bench             Benchmark device X and Y(and more) side-by-side
  -h, --help                             Show this help message
``` 
