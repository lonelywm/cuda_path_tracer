#pragma once
#include <pch.h>
#include "Point.hpp"
#include "Triangle.hpp"
#include "Material.hpp"
#include "BBox.hpp"
#include "BVH.h"



class Mesh {
public:
    Vector<Point>    Points;
    Vector<uint>     Indices;
    Material         Mtrl;
    
public:
    Mesh(CVector<Point>& pts, CVector<uint>& indices, Material& material)
    : Points(pts), Indices(indices), Mtrl(material) 
    {
    }

    void transfer(CVec3f& pos) {
        for(auto& p: Points) {
            p.Pos += pos;
        }
    }

    void scale(CVec3f& s) {
        Vec3f center;
        for (auto& p: Points) {
            center += p.Pos / Points.size();
        }
        for (auto& p: Points) {
            p.Pos = center + (p.Pos - center) * s;
        }
    }

// public:
//     const BoundingBox& getBBox() {
//         if (_bboxDirty) {
//             _bbox = BoundingBox();
//             for (auto& tri: _triangles) {
//                 _bbox.extand(tri.getBBox());
//             }
//             _bboxDirty = false;
//         }
//         return _bbox;
//     }
};