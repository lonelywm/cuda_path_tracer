#include "../Render.h"
#include "../BVH.h"
#include <curand_kernel.h>
#include <fstream>


#undef M_PI
#define M_PI 3.141592653589793f

__device__ float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

__device__ __host__
inline float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

// __global__
// void randInitKernel() {
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     int blockId = blockDim
//     curand_init(idx);
// }

__global__
void rendKernel(Vec3f* buf, BVH* bvh, BVHNode* leafNodes, BVHNode* internalNodes, Vec3f* poss, uint* indices, int width, int height, float fov) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > width * height) return;
    int y = idx / width;
    int x = idx % width;
    float scale = tan(deg2rad(fov * 0.5));           // dep    / height
    float imageAspectRatio = width / (float)height;  // width  / height
	Vec3f eye_pos(278, 273, -800);                   // eye
    float dx = 0;
    float dy = 0;

    // curandState_t randState;
    // curand_init(idx, threadIdx.x, 0, &randState);

    float _x = ( 2 * (x + dx)/(float)width - 1 ) * imageAspectRatio * scale;
    float _y = ( 2 * (y + dy)/(float)height - 1) * scale;
    Vec3f dir(-_x, -_y, 1);
    // dir.x = 0.1;
    // dir.y=0.2;
    // dir.z = 1;
    dir = dir.normalize();
    Ray ray(eye_pos, dir);
    // if (idx == 0) {
    // } else {
    //     return;
    // }

    Intersection isect = bvh->intersect(idx, ray, poss, indices, internalNodes, leafNodes);

    if (isect.happened) {
        buf[idx].x = 1;
        buf[idx].y = isect.t / 1400;
        buf[idx].z = isect.t / 1400;
    } else {
        buf[idx].x = 0;
        buf[idx].y = 0;
        buf[idx].z = 0;
    }

    // if ((idx == 525 || idx == 526)) 
    // printf("IDX:%d dir xyz %f, %f, %f (%f)\n", idx, dir.x, dir.y, dir.z, isect.t);
    // printf("==%4f, %4f, %4f\n", internalNodes[0].BBox.Max.x, internalNodes[0].BBox.Max.y, internalNodes[0].BBox.Max.z);
    // printf("--%4f, %4f, %4f\n", leafNodes[0].BBox.Max.x, leafNodes[0].BBox.Max.y, leafNodes[0].BBox.Max.z);
}

// Render ================================================
void Render::rend(Scene* scene, int width, int height, int spp) {
    _width = width;
    _height = height;
    cudaMalloc((void**)&_buffer, _width * _height * sizeof(Vec3f));
    framebuffer.resize(_width * _height * 12);

    int threadsPerBlock = 256;
    int blocksPerGrid = (width*height + threadsPerBlock - 1) / threadsPerBlock;
    rendKernel<<<blocksPerGrid, threadsPerBlock>>>(_buffer, &scene->_bvh, scene->_bvh._leafNodes, scene->_bvh._internalNodes, scene->_poss, scene->_indices, width, height, 40);
}

__device__
void Render::sampleLight(Vec3f* poss, uint* indices, int numTri, Intersection &pos, float &pdf, curandState_t& randState) const
{
    float emit_area_sum = 0;
    // for (uint32_t i = 0; i < numTri; ++k) {
    //     if (objects[k]->hasEmit()) {
    //         emit_area_sum += objects[k]->Area;
    //     }
    // }
    // float p = get_random_float() * emit_area_sum;  // [0~1]*13650    
    // emit_area_sum = 0;
    // for (uint32_t k = 0; k < objects.size(); ++k) {
    //     if (objects[k]->hasEmit()){
    //         emit_area_sum += objects[k]->Area;
    //         if (p <= emit_area_sum){//random get the first area > p light,return                
    //             objects[k]->Sample(pos, pdf);
    //             break;
    //         }
    //     }
    // }
}

__device__
void Render::shade(const Intersection& isect, const Vec3f& indir) {

}

void Render::output() {
    cudaMemcpy(&framebuffer[0], _buffer, _width * _height * sizeof(Vec3f), cudaMemcpyDeviceToHost);

    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", _width, _height);
    for (auto i = 0; i < _width * _height; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
}
