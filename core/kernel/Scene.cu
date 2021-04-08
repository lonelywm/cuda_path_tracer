#include "../Scene.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include "../TriangleIndices.hpp"

struct minAccessor{
    __host__ __device__
    Vec3f operator () (const BoundingBox& a){
        return a.Min;
    }
};

struct minFunctor{
    __host__ __device__
    Vec3f operator () (const Vec3f& a, const Vec3f& b){
        return minimum(a, b);
    }
};
struct maxAccessor{
    __host__ __device__
    Vec3f operator () (const BoundingBox& a){
        return a.Max;
    }
};

struct maxFunctor{
    __host__ __device__
    Vec3f operator () (const Vec3f& a, const Vec3f& b){
        return maximum(a, b);
    }
};


__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* poss, uint* indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;
    TriangleIndices triangle{indices[idx*3], indices[idx*3 + 1], indices[idx*3 + 2]};
    BBoxs[idx] = BoundingBox(
        poss[triangle.a],
        poss[triangle.b],
        poss[triangle.c]
    );
    auto BBox = BBoxs[idx];

    // printf("Leaf(%d), idx(%02d %02d %02d) min(%0.4f, %0.4f, %0.4f) max(%0.4f, %0.4f, %0.4f) \n", 
    //     idx, indices[idx*3], indices[idx*3 + 1], indices[idx*3 + 2], BBox.Min.x, BBox.Min.y, BBox.Min.z, BBox.Max.x, BBox.Max.y, BBox.Max.z
    // );
}

// For Scene
void Scene::setupDevice() {
    _numPoints = 0;
    _numIndices = 0;
    _numMaterials = 0;
    for (auto* actor: Actors) {
        for (auto* mesh: actor->Meshes) {
            _numPoints += mesh->Points.size();
            _numIndices += mesh->Indices.size();
            _numMaterials += mesh->Materials.size();
        }
    }
    _numTri = _numIndices / 3;
    cudaMalloc(&_poss,    _numPoints*sizeof(Vec3f));
    cudaMalloc(&_indices, _numIndices*sizeof(uint));
    cudaMalloc(&_materials, _numMaterials*sizeof(Material));

    int offsetPos = 0;
    int offsetIdx = 0;
    int offsetMaterial = 0;
    for (auto* actor: Actors) {
        for (auto* mesh: actor->Meshes) {
            Vector<Vec3f> poss;
            for (auto& pt: mesh->Points) { poss.push_back(pt.Pos); }
            Vector<uint> indices;
            for (auto& idx: mesh->Indices) {
                indices.push_back(idx + offsetPos);  // Important and be carefull
            }
            if (mesh->Points.size() > 0) {
                cudaMemcpy(_poss + offsetPos, &poss[0], mesh->Points.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
                offsetPos += mesh->Points.size();
            }
            if (mesh->Indices.size() > 0) {
                cudaMemcpy(_indices + offsetIdx, &indices[0], mesh->Indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
                offsetIdx += mesh->Indices.size();  // Important and be carefull
            }
            if (mesh->Materials.size() > 0) {
                cudaMemcpy(_materials + offsetMaterial, &mesh->Materials[0], mesh->Materials.size() * sizeof(uint), cudaMemcpyHostToDevice);
                offsetMaterial += mesh->Materials.size();
            }
        }
    }
}

void Scene::findMinMax(Vec3f& mMin, Vec3f& mMax) {
    thrust::device_ptr<BoundingBox> dvp(_bboxs);
    mMin = thrust::transform_reduce(dvp, 
            dvp + _numTri,
            minAccessor(), 
            Vec3f(1e9, 1e9, 1e9), 
            minFunctor());
        
    mMax = thrust::transform_reduce(dvp,
            dvp + _numTri,
            maxAccessor(),
            Vec3f(-1e9, -1e9, -1e9),
            maxFunctor());
}

void Scene::computeBoundingBoxes() {
    cudaMalloc(&_bboxs, _numTri*sizeof(BoundingBox));
    int N = _numTri;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // printf("in compute bounding box %d\n", _numTri);
    computeBoundingBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(_numTri, _poss, _indices, _bboxs);
}

void Scene::buildBVH() {
    Vec3f min, max;
    findMinMax(min, max);
    _bvh.setup(_poss, _indices, _bboxs, _numTri, min, max);
}

