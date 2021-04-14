#include <pch.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../Triangle.hpp"
#include "../BVH.h"


__device__
int findSplit(uint64* sortedMortonCodes,
    int first, int last
) {
    // Identical Morton codes => split the range in the middle.
    uint64 firstCode = sortedMortonCodes[first];
    uint64 lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    uint commonPrefix = __clzll(firstCode ^ lastCode);

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
            uint64 splitCode = sortedMortonCodes[newSplit];
            uint splitPrefix = __clzll(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);
    return split;
}

__device__
int2 determineRange(uint64* sortedMortonCodes, int numTriangles, int idx)
{
    //determine the range of keys covered by each internal node (as well as its children)
    //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clzll(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if(idx == 0){
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clzll(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);
    }

    direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
    int min_prefix_range = 0;

    if(idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clzll(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
    }

    int lmax = 2;
    int next_key = idx + lmax*direction;

    while((next_key >= 0) && (next_key <  numTriangles) && (__clzll(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
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
            uint64 Code = sortedMortonCodes[new_val];
            int Prefix = __clzll(sortedMortonCodes[idx] ^ Code);
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

// __device__
// unsigned int expandBits(unsigned int v) {
//     v = (v * 0x00010001u) & 0xFF0000FFu;
//     v = (v * 0x00000101u) & 0x0F00F00Fu;
//     v = (v * 0x00000011u) & 0xC30C30C3u;
//     v = (v * 0x00000005u) & 0x49249249u;
//     return v;
// }

__device__
uint64 expandBits(uint64 v)
{
    assert(v < ((uint64) 1 << 22));
    // _______32 bits_____________32 bits_______
    // |000.............000|000.............vvv| - 22-v significant bits
    // convert to:
    // _______32 bits_____________32 bits_______
    // |v00............00v0|0v...........00v00v|
    v = (v | (v << 16)) & 0x0000003F0000FFFFull; // 0000 0000 0000 0000  0000 0000 0011 1111   0000 0000 0000 0000  1111 1111 1111 1111
    v = (v | (v << 16)) & 0x003F0000FF0000FFull; // 0000 0000 0011 1111  0000 0000 0000 0000   1111 1111 0000 0000  0000 0000 1111 1111
    v = (v | (v <<  8)) & 0x300F00F00F00F00Full; // 0011 0000 0000 1111  0000 0000 1111 0000   0000 1111 0000 0000  1111 0000 0000 1111
    v = (v | (v <<  4)) & 0x30C30C30C30C30C3ull; // 0011 0000 1100 0011  0000 1100 0011 0000   1100 0011 0000 1100  0011 0000 1100 0011
    v = (v | (v <<  2)) & 0x9249249249249249ull; // 1001 0010 0100 1001  0010 0100 1001 0010   0100 1001 0010 0100  1001 0010 0100 1001
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
uint64 morton3D(float x, float y, float z) {
    x = min(max(x * 1048575.0f, 0.0f), 1048575.0f);  // 1048576
    y = min(max(y * 1048575.0f, 0.0f), 1048575.0f);  // 1048576
    z = min(max(z * 1048575.0f, 0.0f), 1048575.0f);  // 1048576
    // x = min(max(x * 1024.0f, 0.0f), 1023.0f);  // 1048576
    // y = min(max(y * 1024.0f, 0.0f), 1023.0f);  // 1048576
    // z = min(max(z * 1024.0f, 0.0f), 1023.0f);  // 1048576
    uint64 xx = expandBits((uint64)x);
    uint64 yy = expandBits((uint64)y);
    uint64 zz = expandBits((uint64)z);
    return xx * 4 + yy * 2 + zz;
}

// ===============================================================

__global__
void computeMortonCodesKernel(uint64* mCodes, uint* objIds, Point* pts, uint* indices, BoundingBox* boxs, int numTri, Vec3f min, Vec3f max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;

    objIds[idx] = idx;
    Vec3f centroid = boxs[idx].getCentroid();
    // Vec3f centroid = (pts[indices[idx*3]].Pos + 
    //     pts[indices[idx*3+1]].Pos + pts[indices[idx*3+2]].Pos) / 3;
    // auto centroid = (centroid0 + centroid1)/2;
    centroid.x = (centroid.x - min.x)/(max.x - min.x);
    centroid.y = (centroid.y - min.y)/(max.y - min.y);
    centroid.z = (centroid.z - min.z)/(max.z - min.z);
    mCodes[idx] = morton3D(centroid.x, centroid.y, centroid.z);
}

// __global__
// void checkMortonCodes(uint64* mCodes, int numTri) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numTri || idx == 0) return;
//     if (idx >= 1) {
//         uint64 pre = mCodes[idx - 1];
//         uint64 now = mCodes[idx];
//         // if (pre == now) 
//         //     printf("omg!!! morton duplicate! [%lu]\n", pre);
//     }
// }



__global__ 
void generateHierarchyKernel(
    uint64* sortedMortonCodes, uint* sorted_object_ids, BVHNode* internalNodes,
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
        if(!Parent->ChildA->BBox.isEmpty() || !Parent->ChildB->BBox.isEmpty())
        {
            Parent->BBox.merge(Parent->ChildA->BBox);
            Parent->BBox.merge(Parent->ChildB->BBox);
            if (Parent->Parent) {
                Parent = Parent->Parent;
                // printf("(%d %d %d [%f %f %f] [%f %f %f])\n", 
                // idx, Parent - internalNodes, Parent->Parent - internalNodes,
                // Parent->BBox.Min.x, Parent->BBox.Min.y, Parent->BBox.Min.z, 
                // Parent->BBox.Max.x, Parent->BBox.Max.y, Parent->BBox.Max.z
                // );
                continue;
            }
        }
        // __syncthreads();
        break;
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
    // if (sorted_object_ids[idx]==4 || sorted_object_ids[idx] == 5) {
    //     printf("SBVH: %d, %d %d; (%f, %f, %f) (%f, %f, %f)\n", idx, sorted_object_ids[idx], leafNodes[idx].ObjectId,
    //     leafNodes[idx].BBox.Min.x, leafNodes[idx].BBox.Min.y, leafNodes[idx].BBox.Min.z, 
    //     leafNodes[idx].BBox.Max.x, leafNodes[idx].BBox.Max.y, leafNodes[idx].BBox.Max.z
    //     );
    // }
}

__global__ 
void logBVHTree(uint* objectIds, uint64* mCodes, BVHNode* internalNodes, BVHNode* leafNodes, int numTri) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTri) return;

    // if (objectIds[idx] == 5) {
    //     Ray ray(Vec3f(278, 273, -800), Vec3f(-0.057365, -0.140683, 0.988391));
    //     BVHNode* parent = &leafNodes[idx];
    //     while(parent) {
    //         printf("IDX[%d] (%f, %f, %f; %f, %f, %f) [%d]\n", parent - internalNodes, 
    //         parent->BBox.Min.x, parent->BBox.Min.y, parent->BBox.Min.z, 
    //         parent->BBox.Max.x, parent->BBox.Max.y, parent->BBox.Max.z,
    //         parent->BBox.intersect(ray)
    //         );
    //         parent=parent->Parent;
    //     }
    //     printf("\n");
    // }

    // if (objectIds[idx]==4) {
    //     BVHNode* node = &leafNodes[idx];
    //     while(node) {
    //         printf("[Idx:%d, ObjId:%d, %d, (%f, %f, %f) (%f, %f, %f)]\n", idx, objectIds[idx], node - internalNodes,
    //             node->BBox.Min.x, node->BBox.Min.y, node->BBox.Min.z,
    //             node->BBox.Max.x, node->BBox.Max.y, node->BBox.Max.z
    //         );
    //         node=node->Parent;
    //     }
    // }

    // if (objectIds[idx]==5) {
    //     BVHNode* node = &leafNodes[idx];
    //     while(node) {
    //         printf("[Idx:%d, ObjId:%d, %d, (%f, %f, %f) (%f, %f, %f)]\n", idx, objectIds[idx], node - internalNodes,
    //             node->BBox.Min.x, node->BBox.Min.y, node->BBox.Min.z,
    //             node->BBox.Max.x, node->BBox.Max.y, node->BBox.Max.z
    //         );
    //         node=node->Parent;
    //     }
    // }

    // if(idx == 0) return;
    // if (mCodes[idx-1] == mCodes[idx]) 
    //     printf("(idx %d; [%l] %f,%f,%f; %f,%f,%f) ", idx, mCodes[idx-1],
    //     leafNodes[idx-1].BBox.Min.x,leafNodes[idx-1].BBox.Min.y,leafNodes[idx-1].BBox.Min.z,
    //     leafNodes[idx].BBox.Max.x,leafNodes[idx].BBox.Max.y,leafNodes[idx].BBox.Max.z
    //     );

    // printf("[%d, %d]", idx, objectIds[idx]);

    // if (!leafNodes[idx].Parent) {
    //     printf("(%d)", objectIds[idx]);
    // }
    // if (idx >= numTri - 1) return;
  
    // if (!internalNodes[idx].ChildA) {
    //     printf("{A%d}", &internalNodes[idx] - &internalNodes[0]);
    // }
    // if (!internalNodes[idx].ChildB) {
    //     printf("{B%d}", &internalNodes[idx] - &internalNodes[0]);
    // }
    // if (!internalNodes[idx].Parent) {
    //     printf("[%d]", &internalNodes[idx] - &internalNodes[0]);
    // }
    
}

// __global__ 
// void streamLogBVHTree(uint* objectIds, uint64* mCodes, BVHNode* internalNodes, BVHNode* leafNodes, int numTri) {
// }

// BVH ==============================================================
void BVH::setup(
    Point* pts, uint* indices, BoundingBox* mBBoxs, int numTriangles, Vec3f min, Vec3f max
) {
    _pts = pts;
    _indices = indices;
    _bboxs = mBBoxs;

    cudaMalloc(&_mortonCodes,   numTriangles*sizeof(uint64));
    cudaMalloc(&_objectIds,     numTriangles*sizeof(unsigned int));
    cudaMalloc(&_leafNodes, (2*numTriangles - 1)*sizeof(BVHNode));
    _internalNodes = _leafNodes + numTriangles;  // So that leafNodes and interNodes are continuity

    int threadsPerBlock = 256;
    int blocksPerGrid = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    // 1.comput morton codes
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        _mortonCodes, _objectIds, _pts, _indices, _bboxs, numTriangles, min, max
    );
    // 2. sort morton codes
    thrust::device_ptr<uint64>       dev_mortonCodes(_mortonCodes);
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
    logBVHTree<<<blocksPerGrid, threadsPerBlock>>>(
        _objectIds, _mortonCodes, _internalNodes, _leafNodes, numTriangles
    );
}

BVH::~BVH(){
    cudaFree(_mortonCodes);
    cudaFree(_objectIds);
    cudaFree(_leafNodes);
    cudaFree(_internalNodes);
}


