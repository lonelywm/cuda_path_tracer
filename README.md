# MPathTracing
A CUDA Path Tracer

# Feature
1. Build BVH-Box in parallel using Morton Code
2. Expand 32-bit Morton Code to 64-bit
3. No material implement, just use diffuse color
4. Very fast compared to using CPU()

# How to build

```
mkdir build
cd build
cmake ..
cmake --build .
```

