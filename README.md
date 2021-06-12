[![tag](https://img.shields.io/github/v/tag/lonelywm/cuda_path_tracer)](https://github.com/lonelywm/ability_system/tags)
[![license](https://img.shields.io/github/license/lonelywm/cuda_path_tracer)](LICENSE) 

![avatar](https://github.com/lonelywm/cuda_path_tracer/blob/master/output/binary1024.png)

## MPathTracing
A CUDA Path Tracer

## Feature
1. Build BVH-Box in parallel using Morton Code
2. Expand 32-bit Morton Code to 64-bit
3. No material implement, just use diffuse color
4. Very fast compared to using CPU

## How to build

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Time cost
| Resolution   | SPP  | Max Reflection   | GPU     |  Time  |
| ------------ | ---- | ---------------- | ------- | ------ |
| 1024x1024    | 128  | 8                | 3060ti  | 29.5s  |
| 512x512      | 64   | 7                | 3060ti  | 3.7s   |
| 256x256      | 32   | 6                | 3060ti  | 0.4s   |

## Compiler compatibility
MSVC >= 19.20