#pragma once
#include <pch.h>
#include <array>
#include "BBox.hpp"
#include "TriangleIndices.hpp"
#include "Intersection.hpp"
#include "Ray.hpp"

struct BVHNode {
    BoundingBox BBox;
    BVHNode* ChildA = nullptr;
    BVHNode* ChildB = nullptr;
    BVHNode* Parent;
    int ObjectId;
    int Flag;
    bool IsLeaf = false;

    __device__
    BVHNode(): Flag(0), ChildA(nullptr), ChildB(nullptr) {
        ObjectId = -1;
    }

    __device__ BVHNode(const BVHNode& other) = default;
    __device__ BVHNode& operator = (BVHNode& other) = default;

    // __device__
    // bool isLeaf() {
    //     return ChildA == nullptr;
    // }
};

class BVH {
public:
    uint64*  _mortonCodes;     // in device
    uint*    _objectIds;       // in device
    BVHNode* _leafNodes;       // in device
    BVHNode* _internalNodes;   // in device
    Point*   _pts;
    uint*    _indices;

    // These are stored in the scene
    int              _numTriangles;
    Vec3f*           _vertices;
    BoundingBox*     _bboxs;
    // Vec3f*           _normals;
    // TriangleIndices* _indices;

public:
    ~BVH();
    void setup (
        Point* poss,
        uint* indices, 
        BoundingBox* mBBoxs, 
        int numTriangles, 
        Vec3f min, 
        Vec3f max
    );
    
    __device__
    Intersection intersect(int idx, const Ray& ray, Point* pts, uint* indices, Geometry* geos, Material* materials, BVHNode* internalNodes, BVHNode* leafNodes) {
        Intersection isect;
        BVHNode nodes[40];
        int ncount = 0;
        nodes[0] = *internalNodes[0].ChildA;
        nodes[1] = *internalNodes[0].ChildB;
        ncount += 2;
        while(ncount > 0) {
            BVHNode& node = nodes[ncount - 1];
            if (node.BBox.intersect(ray)) {
                if (node.IsLeaf) {
                    int index = node.ObjectId;
                    auto tri = Geometry(pts[indices[3*index]].Pos, pts[indices[3*index+1]].Pos, pts[indices[3*index+2]].Pos);
                    auto intersect = tri.intersect(ray);
                    if (intersect.Happened && (!isect.Happened || (intersect.t > 0 && intersect.t < isect.t))  ) {
                        isect = intersect;
                        isect.GeoId = index;

                    //    if (node.ObjectId > 40) {
                    //         printf("#[ObjId: %d [%d] (%f %f %f) (%f %f %f) (%f %f %f)]\n", index, idx,
                    //             pts[indices[3*index]].Pos.x, pts[indices[3*index]].Pos.y, pts[indices[3*index]].Pos.z, 
                    //             pts[indices[3*index+1]].Pos.x, pts[indices[3*index+1]].Pos.y, pts[indices[3*index+1]].Pos.z, 
                    //             pts[indices[3*index+2]].Pos.x, pts[indices[3*index+2]].Pos.y, pts[indices[3*index+2]].Pos.z
                    //         );
                    //     }

                    }
                } else {
                    nodes[ncount]     = *node.ChildB;
                    nodes[ncount - 1] = *node.ChildA;  // replace current node, be very careful.
                    ncount += 2;
                }
            }
            ncount--;
        }
        // printf("\nEND %f\n", isect.t);
        if (isect.Happened) {
            Geometry& geo = geos[isect.GeoId];
            isect.Mtrl = materials[geo.MatId];
        }
        return isect;
    } // intersect

};


