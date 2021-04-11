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