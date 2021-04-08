#pragma once
#include <pch.h>
#include "Point.hpp"
#include "Triangle.hpp"
#include "Material.hpp"
#include "BBox.hpp"
#include "BVH.h"


class MeshD {  // for device
};

class Mesh {
public:
    Vector<Point>    Points;
    Vector<uint>     Indices;
    Vector<Material> Materials;
    
public:
    Mesh(CVector<Point>& pts, CVector<uint>& indices, CVector<Material>& materials)
    : Points(pts), Indices(indices), Materials(materials) 
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