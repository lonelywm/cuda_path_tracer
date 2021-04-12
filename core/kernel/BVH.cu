#include <pch.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../Triangle.hpp"
#include "../BVH.h"


__device__
int findSplit(unsigned int* sortedMortonCodes,
    int first, int last
) {
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);
    return split;
}

__device__
int2 determineRange(uint* sortedMortonCodes, int numTriangles, int idx)
{
    //determine the range of keys covered by each internal node (as well as its children)
    //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if(idx == 0){
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

    }

    direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
    int min_prefix_range = 0;

    if(idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
    }

    int lmax = 2;
    int next_key = idx + lmax*direction;

    while((next_key >= 0) && (next_key <  numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
    {
        lmax *= 2;
        next_key = idx + lmax*direction;
    }
    //find the other end using binary search
    unsigned int l = 0;

    do
    {
        lmax = (lmax + 1) >> 1; // exponential decrease
        int new_val = idx + (l + lmax)*direction ; 

        if(new_val >= 0 && new_val < numTriangles )
        {
            unsigned int Code = sortedMortonCodes[new_val];
            int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
            if (Prefix > min_prefix_range)
                l = l + lmax;
        }
    }
    while (lmax > 1);

    int j = idx + l*direction;

    int left = 0 ; 
    int right = 0;
    
    if(idx < j) {
        left = idx;
        right = j;
    } else {
        left = j;
        right = idx;
    }

    // printf("idx : (%d) returning range (%d, %d) \n" , idx , left, right);
    return make_int2(left,right);
}

__device__
unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
unsigned int morton3D(float x, float y, float z) {
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

// ===============================================================

__global__
void computeMortonCodesKernel(uint* mCodes, uint* objIds, BoundingBox* boxs, int numTri, Vec3f min, Vec3f max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;

    objIds[idx] = idx;
    Vec3f centroid = boxs[idx].getCentroid();
    centroid.x = (centroid.x - min.x)/(max.x - min.x);
    centroid.y = (centroid.y - min.y)/(max.y - min.y);
    centroid.z = (centroid.z - min.z)/(max.z - min.z);
    mCodes[idx] = morton3D(centroid.x, centroid.y, centroid.z);
}

// __global__
// void checkMortonCodes(uint* mCodes, int numTri) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numTri || idx == 0) return;

//     if (idx >= 1) {
//         uint pre = mCodes[idx - 1];
//         uint now = mCodes[idx];
//         if (pre == now) 
//             printf("omg!!! morton duplicate!\n");
//     }
// }



__global__ 
void generateHierarchyKernel(
    uint* sortedMortonCodes, uint* sorted_object_ids, BVHNode* internalNodes,
    BVHNode* leafNodes, int numTri, BoundingBox* BBoxs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri - 1) return;

    internalNodes[idx].IsLeaf = false;

    int2 range = determineRange(sortedMortonCodes, numTri, idx);
    int first = range.x;
    int last = range.y;

    int split = findSplit(sortedMortonCodes, first, last);
    BVHNode* childA;

    bool isLeafA = false;
    bool isLeafB = false;
    if (split == first) {
        childA = &leafNodes[split];
        isLeafA = true;
    }
    else {
        childA = &internalNodes[split];
    }

    BVHNode* childB;
    if (split + 1 == last) {
        childB = &leafNodes[split + 1];
        isLeafB = true;
    } else {
        childB = &internalNodes[split + 1];
    }


    childA->Parent = &internalNodes[idx];
    childB->Parent = &internalNodes[idx];
    internalNodes[idx].ChildA = childA;
    internalNodes[idx].ChildB = childB;

}

__global__
void infoBBoxesKernel(BVHNode* leafNodes, BVHNode* internalNodes, int numTriangles) {
    BVHNode* nodes = leafNodes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;
    
    BVHNode* node = &nodes[idx];

}

__global__ 
void computeBBoxesKernel(BVHNode* leafNodes, BVHNode* internalNodes, int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    BVHNode* Parent = leafNodes[idx].Parent;

    while(Parent)
    {
        if(!Parent->ChildA->BBox.isEmpty() && !Parent->ChildB->BBox.isEmpty())
        {
            Parent->BBox.merge(Parent->ChildA->BBox);
            Parent->BBox.merge(Parent->ChildB->BBox);
            Parent = Parent->Parent;
        } else{
            break;
        }
    }
}

__global__ 
void setupLeafNodesKernel(uint* sorted_object_ids, BVHNode* leafNodes, BoundingBox* bboxes, int numTri) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;
    int ObjectId = leafNodes[idx].ObjectId = sorted_object_ids[idx];
    leafNodes[idx].ChildA = nullptr;
    leafNodes[idx].ChildB = nullptr;
    leafNodes[idx].IsLeaf = true;
    leafNodes[idx].BBox   = bboxes[ObjectId];
}

__global__ 
void printfMortonCodes(uint* mortonCodes, int count) {
    for (int i=0; i<count; i++) {
    }
}

// BVH ==============================================================
void BVH::setup(
    Point* pts, uint* indices, BoundingBox* mBBoxs, int numTriangles, Vec3f min, Vec3f max
) {
    _pts = pts;
    _indices = indices;
    _bboxs = mBBoxs;

    cudaMalloc(&_mortonCodes,   numTriangles*sizeof(unsigned int));
    cudaMalloc(&_objectIds,     numTriangles*sizeof(unsigned int));
    cudaMalloc(&_leafNodes,     numTriangles*sizeof(BVHNode));
    cudaMalloc(&_internalNodes, (numTriangles - 1)*sizeof(BVHNode));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

    // 1.comput morton codes
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        _mortonCodes, _objectIds, _bboxs, numTriangles, min, max
    );

    // 2. sort morton codes
    thrust::device_ptr<unsigned int> dev_mortonCodes(_mortonCodes);
    thrust::device_ptr<unsigned int> dev_object_ids(_objectIds);
    thrust::sort_by_key(dev_mortonCodes, dev_mortonCodes + numTriangles, dev_object_ids);

    // checkMortonCodes<<<blocksPerGrid, threadsPerBlock>>>(
    //     _mortonCodes, numTriangles
    // );

    // 3. build tree
    setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        _objectIds, _leafNodes, _bboxs, numTriangles
    );
    generateHierarchyKernel<<<blocksPerGrid, threadsPerBlock>>>(
        _mortonCodes, _objectIds, _internalNodes, _leafNodes, numTriangles, _bboxs
    );

    // printfMortonCodes<<<1, 1>>>(_mortonCodes, numTriangles);
    // infoBBoxesKernel<<<blocksPerGrid, threadsPerBlock>>>(_leafNodes, _internalNodes, numTriangles);

    computeBBoxesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        _leafNodes, _internalNodes, numTriangles
    );

    // Vector<BVHNode> inters;
    // inters.resize(numTriangles-1);
    // cudaMemcpy(&inters[0], _internalNodes, (numTriangles-1) * sizeof(BVHNode), cudaMemcpyDeviceToHost);
    // printf("...");
}

BVH::~BVH(){
    cudaFree(_mortonCodes);
    cudaFree(_objectIds);
    cudaFree(_leafNodes);
    cudaFree(_internalNodes);
}


