#pragma once
#include <pch.h>
#include "Material.hpp"

struct Intersection {
    float t;
    Vec3f N;
    Vec3f bary;
    Vec3f Pos;
    int objectIdx;
    Material material; // For smooth shading
    bool happened = false;

    __device__ Intersection() {
        happened = false;
        t = 0;
    }
};
