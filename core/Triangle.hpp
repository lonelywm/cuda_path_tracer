#pragma once
#include <pch.h>
#include "Geometry.hpp"
#include "Point.hpp"
#include "BBox.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include <curand_kernel.h>


struct Triangle: public Geometry {
    Point Vs[3];
    Vec3f Nor;
    Vec3f E1, E2;  // p1-p0, p2-p0;
    float Area;

    __host__ __device__
    Triangle(CVec3f& p0, CVec3f& p1, CVec3f& p2) {
        Vs[0].Pos = p0;
        Vs[1].Pos = p1;
        Vs[2].Pos = p2;
        E1 = p1 - p0;
        E2 = p2 - p0;
        Nor = E1.cross(E2);
        Area = Nor.norm()*0.5f;
        Nor = Nor.normalize();
    }

    __device__
    Intersection intersect(const Ray& ray) {
        Intersection inter;
        if (ray.Dir * Nor > 0) {
            return inter;
        }

        float u, v, t_tmp = 0;
        Vec3f pvec = ray.Dir.cross(E2);

        float det = E1*pvec;
        if (fabs(det) < EPSILON)
            return inter;

        float det_inv = 1. / det;
        Vec3f tvec = ray.Pos - Vs[0].Pos;
        u = (tvec * pvec) * det_inv;
        if (u < 0 || u > 1)
            return inter;
        
        Vec3f qvec = tvec.cross(E1);
        v = (ray.Dir * qvec) * det_inv;
        if (v < 0 || u + v > 1)
            return inter;
        t_tmp = (E2 * qvec) * det_inv;

        // TODO find ray triangle intersection
        if (t_tmp <= 0)
            return inter;

        inter.happened = true;
        // inter.coords = ray(t_tmp);
        // inter.m = this->m;
        // inter.normal = normal;
        inter.t = t_tmp;
        // inter.objectId = 
        return inter;
    }

    __device__
    void sample(Intersection& isect, float &pdf, curandState& randState) {
        float2 xy = curand_normal2(&randState);
        float x = xy.x;
        float y = xy.y;
        isect.Pos = Vs[0].Pos * (1.0f - x) + Vs[1].Pos * (x * (1.0f - y)) + Vs[2].Pos * (x * y);
        isect.N = Nor;
        pdf = 1.0f / Area;
    }
};