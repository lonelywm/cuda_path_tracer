#pragma once
#include <pch.h>
#include <Ray.hpp>



struct BoundingBox {
public:
    Vec3f Min, Max;
    bool Inited = false;
    bool dirty = true;

    __host__ __device__
    BoundingBox(): Inited(true) {
        Real minNum = std::numeric_limits<Real>::lowest();
        Real maxNum = std::numeric_limits<Real>::max();
        Max = Vec3f(minNum, minNum, minNum);
        Min = Vec3f(maxNum, maxNum, maxNum);
    }
    
    __host__ __device__
    BoundingBox(CVec3f& p): Min(p), Max(p), Inited(true) {}

    __host__ __device__
    BoundingBox(CVec3f& p0, CVec3f& p1, CVec3f& p2): Min(p0), Max(p0), Inited(true) {
        merge(p1);
        merge(p2);
    }

    __host__ __device__ __inline__
    bool isEmpty() {
        return !Inited;
    }

    __host__ __device__ __inline__
    Vec3f getCentroid() {
        return (Min + Max) / 2.0f;
    }
    
    __host__ __device__ __inline__
    void merge(CVec3f& p) {
        if (!Inited) {
            *this = p;
            Inited = true;
            return;
        }
        if (p.x < Min.x) Min.x = p.x;
        if (p.x > Max.x) Max.x = p.x;
        if (p.y < Min.y) Min.y = p.y;
        if (p.y > Max.y) Max.y = p.y;
        if (p.z < Min.z) Min.z = p.z;
        if (p.z > Max.z) Max.z = p.z;
        dirty = true;
    }
    
    __host__ __device__ __inline__
    void merge(const BoundingBox& bbox) {
        if (!bbox.Inited) return;
        if (!Inited) {
            *this = bbox;
            Inited = true;
            return;
        }
        if (bbox.Min.x < Min.x) Min.x = bbox.Min.x;
        if (bbox.Max.x > Max.x) Max.x = bbox.Max.x;
        if (bbox.Min.y < Min.y) Min.y = bbox.Min.y;
        if (bbox.Max.y > Max.y) Max.y = bbox.Max.y;
        if (bbox.Min.z < Min.z) Min.z = bbox.Min.z;
        if (bbox.Max.z > Max.z) Max.z = bbox.Max.z;
        dirty = true;
    }

    __host__ __device__
    inline bool intersect(const Ray& ray, bool debug = false) const {
        const auto& origin = ray.Pos;
        float tEnter = std::numeric_limits<Real>::lowest();
        float tExit  = std::numeric_limits<Real>::max();
        for (int i = 0; i < 3; i++)
        {
            float min = (Min[i] - origin[i]) / ray.Dir[i];
            float max = (Max[i] - origin[i]) / ray.Dir[i];

            // Fix Zero
            if (ray.Dir[i]== 0) {
                min = (Min[i] - origin[i] <= 0) ? std::numeric_limits<Real>::lowest() : std::numeric_limits<Real>::max();
                max = (Max[i] - origin[i] >= 0) ? std::numeric_limits<Real>::max() : std::numeric_limits<Real>::lowest();
            }
     
            if (ray.Dir[i] < 0) {
                // swap min max
                tEnter = max > tEnter ? max : tEnter;
                tExit  = min < tExit  ? min : tExit;
            } else {
                tEnter = min > tEnter ? min : tEnter;
                tExit  = max < tExit  ? max : tExit;
            }

        }

        return tEnter <= tExit && tExit >= 0;
    }
};