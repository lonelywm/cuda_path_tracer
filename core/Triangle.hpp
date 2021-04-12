#pragma once
#include <pch.h>
#include "Geometry.hpp"
#include "Point.hpp"
#include "BBox.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include <curand_kernel.h>


struct Geometry {
    Point Vs[3];
    Vec3f Nor;
    Vec3f E1, E2;  // p1-p0, p2-p0;
    float Area;
    int  MatId = -1;
    bool Emit;
    int3 Indices;
    bool dirty = true;
    
    __host__ __device__
    Geometry(uint i0, uint i1, uint i2) {
        Indices.x = i0;
        Indices.y = i1;
        Indices.z = i2;
        dirty = true;
        MatId = -1;
    }

    __host__ __device__
    Geometry(CVec3f& p0, CVec3f& p1, CVec3f& p2) {
        Vs[0].Pos = p0;
        Vs[1].Pos = p1;
        Vs[2].Pos = p2;
        E1 = p1 - p0;
        E2 = p2 - p0;
        Nor = E1.cross(E2);
        Area = Nor.norm()*0.5f;
        Nor = Nor.normalize();
        MatId = -1;
        Emit = false;
        dirty = false;
    }

    __host__ __device__ __inline__
    Geometry& update(Point* pts, Material* mats) {
        if (!dirty) return *this;
        Vs[0] = pts[Indices.x];
        Vs[1] = pts[Indices.y];
        Vs[2] = pts[Indices.z];
        Emit = false;

        // printf("OnUpdate: Dirty:%d, Indices(%d, %d, %d) MatId: %d, Emi: %d\n", dirty, Indices.x, Indices.y, Indices.z, MatId, mats[MatId].hasEmission());
        
        if (MatId >= 0 && mats[MatId].hasEmission()) {
            Emit = true;
        }
        E1 = Vs[1].Pos - Vs[0].Pos;
        E2 = Vs[2].Pos - Vs[0].Pos;
        Nor = E1.cross(E2);
        Area = Nor.norm()*0.5f;
        Nor = Nor.normalize();
        dirty = false;
        return *this;
    }

    // __host__ __device__
    // Geometry(Point& p0, Point& p1, Point& p2, Material* materials) {
    //     Vs[0] = p0;
    //     Vs[1] = p1;
    //     Vs[2] = p2;
    //     E1 = p1 - p0;
    //     E2 = p2 - p0;
    //     Nor = E1.cross(E2);
    //     Area = Nor.norm()*0.5f;
    //     Nor = Nor.normalize();
    // }

    __device__ __inline__
    Intersection intersect(const Ray& ray) {
        Intersection inter;
        if (ray.Dir.dot(Nor) > 0) {
            return inter;
        }

        float u, v, t_tmp = 0;
        Vec3f pvec = ray.Dir.cross(E2);

        float det = E1.dot(pvec);
        if (fabs(det) < EPSILON)
            return inter;

        float det_inv = 1. / det;
        Vec3f tvec = ray.Pos - Vs[0].Pos;
        u = tvec.dot(pvec) * det_inv;
        if (u < 0 || u > 1)
            return inter;
        
        Vec3f qvec = tvec.cross(E1);
        v = ray.Dir.dot(qvec) * det_inv;
        if (v < 0 || u + v > 1)
            return inter;
        t_tmp = E2.dot(qvec) * det_inv;

        // TODO find ray triangle intersection
        if (t_tmp <= 0)
            return inter;

        inter.Pos = ray.Pos + ray.Dir * t_tmp;
        inter.Happened = true;
        inter.t = t_tmp;
        inter.N = Nor;
        return inter;
    }

    __device__ __inline__
    void sample(Intersection& isect, float &pdf, curandState& randState) {
        float x = curand_uniform(&randState);
        float y = curand_uniform(&randState);
        isect.Pos = Vs[0].Pos * (1.0f - x) + Vs[1].Pos * (x * (1.0f - y)) + Vs[2].Pos * (x * y);
        isect.N = Nor;
        pdf = 1.0f / Area;
    }

};