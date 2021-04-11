#pragma once
#include <pch.h>
#include "Material.hpp"

struct Intersection {
    float t;
    Vec3f N;
    Vec3f Pos;
    Material Mtrl;
    uint GeoId;
    bool Happened = false;

    __device__ 
    Intersection() {
        Happened = false;
        t = 0;
    }

    __host__ __device__ __inline__
    float distance(Vec3f dir) {
        Vec3f d = t*dir;
        return d.norm();
    }

};
