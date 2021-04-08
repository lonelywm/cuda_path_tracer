#pragma once
#include <pch.h>
#include <Actor.hpp>

class Scene {
public:
    Vector<Actor*> Actors;

public:
    Vec3f* _poss;
    uint* _indices;
    Material* _materials;

    int _numPoints  = 0;
    int _numIndices = 0;
    int _numMaterials = 0;
    int _numTri     = 0; // triangle = _numIndices / 3
    // Material* materials;
    BoundingBox* _bboxs; //Per Triangle Bounding Box
    BVH _bvh;

public:
    void setupDevice();
    void computeBoundingBoxes();
    void buildBVH();
    
private:
    void findMinMax(Vec3f& mMin, Vec3f& mMax);
};