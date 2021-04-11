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
void computeBoundingBoxes_kernel(int numTriangles, Point* pts, uint* indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;
    // TriangleIndices triangle{indices[idx*3], indices[idx*3 + 1], indices[idx*3 + 2]};
    BBoxs[idx] = BoundingBox(
        pts[indices[idx*3]].Pos,
        pts[indices[idx*3+1]].Pos,
        pts[indices[idx*3+2]].Pos
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
            _numMaterials += 1;
        }
    }
    _numTri = _numIndices / 3;
    cudaMalloc(&_pts,       _numPoints*sizeof(Point));
    cudaMalloc(&_indices,   _numIndices*sizeof(uint));
    cudaMalloc(&_materials, _numMaterials*sizeof(Material));
    cudaMalloc(&_geos,      _numTri*sizeof(Geometry));

    int offsetPos = 0;
    int offsetIdx = 0;
    int offsetMaterial = 0;
    int offsetGeo = 0;
    for (auto* actor: Actors) {
        for (auto* mesh: actor->Meshes) {
            Vector<Point> pts;
            for (auto pt: mesh->Points) { 
                pts.push_back(pt);
            }
            Vector<uint> indices;
            Vector<Geometry> geos;
            for (auto& idx: mesh->Indices) {
                indices.push_back(idx + offsetPos);  // Important and be carefull
            }
            for (int i=0; i<mesh->Indices.size(); i+=3) {
                Geometry geo(offsetPos + mesh->Indices[i], offsetPos + mesh->Indices[i+1], offsetPos + mesh->Indices[i+2]);
                geo.MatId = offsetMaterial;
                geos.push_back(geo);
            }
            if (mesh->Points.size() > 0) {
                cudaMemcpy(_pts + offsetPos, &pts[0], mesh->Points.size() * sizeof(Point), cudaMemcpyHostToDevice);
                offsetPos += mesh->Points.size();
            }
            if (mesh->Indices.size() > 0) {
                cudaMemcpy(_indices + offsetIdx, &indices[0], mesh->Indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
                offsetIdx += mesh->Indices.size();  // Important and be carefull
            }
            if (true) {
                cudaMemcpy(_materials + offsetMaterial, &mesh->Mtrl, sizeof(Material), cudaMemcpyHostToDevice);
                offsetMaterial += 1;
            }
            if (geos.size() > 0) {
                cudaMemcpy(_geos + offsetGeo, &geos[0], geos.size() * sizeof(Geometry), cudaMemcpyHostToDevice);
                offsetGeo += geos.size();
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
    computeBoundingBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(_numTri, _pts, _indices, _bboxs);
}

void Scene::buildBVH() {
    Vec3f min, max;
    findMinMax(min, max);
    _bvh.setup(_pts, _indices, _bboxs, _numTri, min, max);
}

