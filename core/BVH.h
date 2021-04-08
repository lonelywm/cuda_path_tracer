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
    uint*    _mortonCodes;     // in device
    uint*    _objectIds;       // in device
    BVHNode* _leafNodes;       // in device
    BVHNode* _internalNodes;   // in device
    Vec3f*   _poss;
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
        Vec3f* poss, uint* indices,
        BoundingBox* mBBoxs, int numTriangles, Vec3f min, Vec3f max
    );

        
    __device__
    Intersection intersect(int idx, const Ray& ray, Vec3f* poss, uint* indices, BVHNode* internalNodes, BVHNode* leafNodes) {
        Intersection isect;
        // if (!(idx == 565)) return isect;

        BVHNode nodes[48];
        int ncount = 0;
        nodes[0] = *internalNodes[0].ChildA;
        nodes[1] = *internalNodes[0].ChildB;
        ncount += 2;
        int loop = 0;
        // printf("%4f, %4f, %4f", nodes[0].BBox.Max.x, nodes[0].BBox.Max.y, nodes[0].BBox.Max.z);
        while(ncount > 0) {
            BVHNode& node = nodes[ncount - 1];
            loop++;

            // printf("\n{ %d, %d, %d } [BOX]\n: max(%f, %f, %f) min(%f, %f, %f)[%d]\n\n", 
            //     loop, idx, ncount,
            //     node.BBox.Max.x, node.BBox.Max.y, node.BBox.Max.z,
            //     node.BBox.Min.x, node.BBox.Min.y, node.BBox.Min.z,
            //     node.BBox.intersect(ray, true)
            // );

            if (node.BBox.intersect(ray)) {
                // printf(
                //     "+idx:%d, %d, %d, %d, %d, Max(%f, %f, %f) Min(%f, %f, %f) \n", 
                //     idx, ncount, node.IsLeaf, node.ChildA->IsLeaf, node.ChildB->IsLeaf,
                //     node.BBox.Max.x, node.BBox.Max.y, node.BBox.Max.z,
                //     node.BBox.Min.x, node.BBox.Min.y, node.BBox.Min.z
                // );
                // printf("#[%d]", node.IsLeaf);
                if (node.IsLeaf) {
                    // printf("*");
                    int index = node.ObjectId;
                    auto tri = Triangle(poss[indices[3*index]], poss[indices[3*index+1]], poss[indices[3*index+2]]);
                    auto intersect = tri.intersect(ray);
                    // printf(
                    //     "P0(%f, %f, %f) P1(%f, %f, %f) P2(%f, %f, %f)"
                    //     "Ori(%f, %f, %f) Dir(%f, %f, %f)\n"
                    //     "Happened(%d) Index(%d, %d %d %d)\n",
                    //     poss[3*index].x, poss[3*index].y, poss[3*index].z,
                    //     poss[3*index+1].x, poss[3*index+1].y, poss[3*index+1].z,
                    //     poss[3*index+2].x, poss[3*index+2].y, poss[3*index+2].z,
                    //     ray.Pos.x, ray.Pos.y, ray.Pos.z,
                    //     ray.Dir.x, ray.Dir.y, ray.Dir.z,
                    //     intersect.happened, index, 3*index, 3*index+1, 3*index+2
                    // );

                    if (intersect.happened && (!isect.happened || (intersect.t > 0 && intersect.t < isect.t))  ) {
                        isect = intersect;
                        // printf("$ <<<%f>>>", intersect.t);
                    }
                } else {
                    nodes[ncount]     = *node.ChildB;
                    nodes[ncount - 1] = *node.ChildA;  // replace current node, be very careful.
                    ncount += 2;
                    // printf("@");
                }
                // printf("[END]\n");
            }

            ncount--;
        }
        // printf("\nEND %f\n", isect.t);
        return isect;
    } // intersect



    // void computeMortonCodesKernel(uint* mCodes, uint* objIds, BoundingBox* boxs, int numTri, Vec3f min, Vec3f max);
    // void setupLeafNodesKernel();
    // void computeBBoxesKernel(BVHNode* leafNodes, BVHNode* internalNodes, int numTriangles);
};


