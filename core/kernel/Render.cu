#include "../Render.h"
#include "../BVH.h"
#include <curand_kernel.h>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>


__device__ float deg2rad(const float& deg) { return deg * PI / 180.0; }

__device__ __host__
inline float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

__device__
void sampleLight(
    Intersection &isect, float &pdf, Point* pts, Geometry* geos, Material* materials,
    float* emitAreas, float& emitAreaSum, uint emitAreaNum, uint* emitAreaIds, curandState_t& randState
) {
    float p = curand_uniform(&randState) * emitAreaSum;
    float start = 0;
    
    // find whice Geometry (index) to sample from
    uint index = -1;
    for (int i=0; i<emitAreaNum; i++) {
        start += emitAreas[i];
        if (start >= p) {
            index = emitAreaIds[i];
            break;
        }
    }

    // printf("%d %d %d %d\n", emitAreaIds[0], emitAreaIds[1], emitAreaIds[2], emitAreaIds[3]);
    // printf("(AL: %f, P: %f, S: %f ,0: %f, I:%d) \n", emitAreaSum, p, start, emitAreas[0], index);

    if (index != -1) {
        geos[index].update(pts, materials).sample(isect, pdf, randState);
        int MatId = geos[index].MatId;
        if (MatId >= 0) {
            isect.Mtrl = materials[MatId];
        }
    }
    
}


__device__
void sampleAveCenterLight(
    Intersection &isect, float &pdf, Point* pts, Geometry* geos, Material* materials,
    float* emitAreas, float& emitAreaSum, uint emitAreaNum, uint* emitAreaIds
) {
    pdf = 0;
    Intersection isection;
    float pdfTmp = 0;
    for (int i = 0; i < emitAreaNum; i++) {
        uint index = emitAreaIds[i];
        geos[index].update(pts, materials).sampleCenter(isection, pdfTmp);
        int MatId = geos[index].MatId;
        if (MatId >= 0) {
            isect.Mtrl = materials[MatId];
        }
        isect.Pos += isection.Pos / emitAreaNum;
        isect.N = isection.N;
        isect.t = isection.t / emitAreaNum;
        pdf += pdfTmp / emitAreaNum;
        isect.Happened = true;
    }
}

__global__
void calculateLightArea(
    float* emitAreas, uint* emitAreaIds, int triangles, Point* pts, uint* indices,
    Geometry* geos, Material* materials
) {
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= triangles) return;

    emitAreaIds[idx] = idx;
    Geometry& geo = geos[idx];
    geo.update(pts, materials);
    emitAreas[idx] = geo.Emit * geo.Area;
    // printf("init Light %d (%f) N: %f, Emit: %d\n", idx, emitAreas[idx], geo.Area, geo.Emit);
    // TODO. 优化规约求和
}


__global__
void phongKernel(
    Vec3f* buf, BVH* bvh, BVHNode* leafNodes, BVHNode* internalNodes,
    Point* pts, uint* indices, Geometry* geos, Material* materials,
    float* emitAreas, float emitAreaSum, uint emitAreaNum, uint* emitAreaIds,
    int width, int height, float fov, int maxTraceDepth, bool onlyDirectLight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height) return;
    int idx = y * width + x;
    const float EPS = 0.0001f;
    float scale = tan(deg2rad(fov * 0.5));           // dep    / height
    float imageAspectRatio = width / (float)height;  // width  / height
	Vec3f eye_pos(278, 273, -800);                   // eye
    float dx = 0.5;
    float dy = 0.5;
    float _x = ( 2 * (x + dx)/(float)width - 1 ) * imageAspectRatio * scale;
    float _y = ( 2 * (y + dy)/(float)height - 1) * scale;
    Vec3f dir(_x, -_y, 1);

    dir = dir.normalize();
    auto ray = Ray(eye_pos, dir);
    Intersection isectObj = bvh->intersect(idx, ray, pts, indices, geos, materials, internalNodes, leafNodes);

    Vec3f color;
    if (isectObj.Happened) {
        color = isectObj.Mtrl.kd * 0.1;
        float pdf = 300;
        float cos = isectObj.N.dot(-dir);
        if (cos < 0) cos = 0;
        float ke = 1000;
        float r2 = isectObj.distance(ray.Dir) * isectObj.distance(ray.Dir);
        Vec3f c = isectObj.Mtrl.kd * 2500 / r2 * cos * pdf;
        color += c;
    }
    buf[idx] = color;
}


__global__
void renderKernel(
    Vec3f* buf, BVH* bvh, BVHNode* leafNodes, BVHNode* internalNodes,
    Point* pts, uint* indices, Geometry* geos, Material* materials,
    float* emitAreas, float emitAreaSum, uint emitAreaNum, uint* emitAreaIds,
    int width, int height, float fov, int maxTraceDepth, bool onlyDirectLight
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int dpp = blockDim.x;
    __shared__ Vec3f pixels[MAX_DPP];

    const float EPS = 0.0001f;

    if (idx >= width * height) return;
    int y = idx / width;
    int x = idx % width;
    float scale = tan(deg2rad(fov * 0.5));           // dep    / height
    float imageAspectRatio = width / (float)height;  // width  / height
	Vec3f eye_pos(278, 273, -800);                   // eye
    

    curandState_t randState;
    curand_init(idx * blockDim.x + tid, tid, 0, &randState);

    float dx = curand_uniform(&randState);
    float dy = curand_uniform(&randState);

    float _x = ( 2 * (x + dx)/(float)width - 1 ) * imageAspectRatio * scale;
    float _y = ( 2 * (y + dy)/(float)height - 1) * scale;
    Vec3f dir(_x, -_y, 1);

    // 用一组变量来将递归变成循环
    // 保留每一步结果的ray,乘法系数,递归深度,是否结束等
    Ray rays[MAX_TRACE_DEPTH+1];
    int raysNum = 0;
    Vec3f alphas[MAX_TRACE_DEPTH+1];
    Vec3f beta[MAX_TRACE_DEPTH + 1];
    bool feedback = false;

    {
        dir = dir.normalize();
        rays[raysNum] = Ray(eye_pos, dir);
        raysNum++;
    }

    pixels[tid] = 0;
    while (!feedback && raysNum < maxTraceDepth) {
        Ray& ray = rays[raysNum - 1];
        Intersection isectObj = bvh->intersect(idx, ray, pts, indices, geos, materials, internalNodes, leafNodes);
        feedback = true;
        if (isectObj.Happened) {
            if (isectObj.Mtrl.hasEmission()) {
                beta[raysNum-1] = isectObj.Mtrl.ke;
            }
            else {
                // 直接光照
                Vec3f Lo_dir;
                {
                    float lightPdf;
                    Intersection isectSample;
                    sampleLight(isectSample, lightPdf, pts, geos, materials, emitAreas, emitAreaSum, emitAreaNum, emitAreaIds, randState);
                    Vec3f obj2Light = isectSample.Pos - isectObj.Pos;
                    Vec3f obj2LightDir = obj2Light.normalize();

                    // 光线是否被遮挡(从采样点发出一条光线，然后当成射线头像物体的交点，查看交点距离与物体到光采样点的距离)
                    Ray dirRay(isectObj.Pos, obj2LightDir);
                    auto lightIsect = bvh->intersect(idx, dirRay, pts, indices, geos, materials, internalNodes, leafNodes);

                    float dist = lightIsect.distance(obj2LightDir);
                    float norm = obj2Light.norm();
                    if (lightIsect.distance(obj2LightDir) - obj2Light.norm() > -EPS)
                    {
                        //     // Vec f_r = isectObj.Mtrl->eval(obj2LightDir, wo, hit_obj.normal);
                        Vec3f f_r = isectObj.Mtrl.eval(obj2LightDir, -ray.Dir, isectObj.N);
                        float r2 = obj2Light.norm2();
                        float cosA = isectObj.N.dot(obj2LightDir);
                        float cosB = isectSample.N.dot(-obj2LightDir);
                        if (cosA < 0) cosA = 0;
                        if (cosB < 0) cosB = 0;
                        Lo_dir = isectSample.Mtrl.ke * f_r * cosA * cosB / r2 / lightPdf;
                    }
                }
                beta[raysNum - 1] = Lo_dir;
                if (onlyDirectLight) break;

                // 间接光照
                Vec3f Lo_indir;
                {
                    if (curand_uniform(&randState) < RussianRoulette)
                    {
                        // Geometry& hitObj = geos[isectObj.GeoId];
                        // 在此物体上采样（按方向）
                        Vec3f sampleDir = isectObj.Mtrl.sample(isectObj.N, randState);
                        float pdf = isectObj.Mtrl.pdf(sampleDir, isectObj.N);
                        if (pdf > EPS) {
                            Ray toNext(isectObj.Pos, sampleDir);
                            Intersection nextObjIsect = bvh->intersect(idx, toNext, pts, indices, geos, materials, internalNodes, leafNodes);
                            if (nextObjIsect.Happened && nextObjIsect.Mtrl.ke.norm2() <= 0) {  // !!!ke.norm2() <= 0
                                Vec3f fr = isectObj.Mtrl.eval(sampleDir, -ray.Dir, isectObj.N);
                                float cos = sampleDir.dot(isectObj.N);
                                if (cos <= 0) cos = 0;
                                Vec3f alpha = fr * cos / pdf / RussianRoulette;
                                alphas[raysNum - 1] = alpha;
                                rays[raysNum] = Ray(isectObj.Pos, sampleDir);
                                raysNum++;
                                feedback = false;
                                continue;
                            }
                        }
                    }
                }


            }
        }  // if happend
    }  // end for


    Vec3f result;
    while (raysNum > 0) {
        result = result * alphas[raysNum-1] + beta[raysNum-1];
        raysNum--;
    }
    pixels[tid] = result;

    __syncthreads();
    for (int stride = dpp/2; stride>0; stride/=2) {
        if (tid < stride) pixels[tid] += pixels[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        buf[idx] = pixels[0] / dpp;
        // if (idx == 47292) {
        //     buf[idx] = Vec3f(0, 0, 1);
        // }
    }
}

// Render ================================================
void Render::init(Scene* scene, int width, int height, int spp, int maxTraceDepth, bool onlyDirectLight) {
    _width = width;
    _height = height;
    _scene = scene;
    _spp = spp;
    _maxTraceDepth = maxTraceDepth;
    _onlyDirectLight = onlyDirectLight;

    cudaMalloc((void**)&_buffer, _width * _height * sizeof(Vec3f));
    framebuffer.resize(_width * _height);

    cudaMalloc((void**)&_emitAreas, scene->_numTri * sizeof(float));
    cudaMalloc((void**)&_emitAreaIds, scene->_numTri * sizeof(uint));

    int threadsPerBlock = 256;
    int blocksPerGridForGeometry = (scene->_numTri + threadsPerBlock - 1) / threadsPerBlock;
    calculateLightArea<<<blocksPerGridForGeometry, threadsPerBlock>>>(_emitAreas, _emitAreaIds, scene->_numTri, scene->_pts,
        scene->_indices, scene->_geos, scene->_materials);

    thrust::device_ptr<float> dev_emitAreas(_emitAreas);
    thrust::device_ptr<unsigned int> dev_emitAreaIds(_emitAreaIds);
    thrust::sort_by_key(dev_emitAreas, dev_emitAreas + scene->_numTri, dev_emitAreaIds, thrust::greater<float>());

    Vector<float> buf(scene->_numTri);
    cudaMemcpy(&buf[0], _emitAreas, scene->_numTri * sizeof(float), cudaMemcpyDeviceToHost);

    _emitAreaSum = 0;
    for (int i=0; i<scene->_numTri; i++) {
        _emitAreaSum += buf[i];
        if (buf[i] > 0) _emitAreaNum++;
        else break;
    }
}

void Render::render() {
    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    renderKernel<<<_width*_height, _spp>>>(
        _buffer, &_scene->_bvh, _scene->_bvh._leafNodes, _scene->_bvh._internalNodes,
        _scene->_pts, _scene->_indices, _scene->_geos, _scene->_materials,
        _emitAreas, _emitAreaSum, _emitAreaNum, _emitAreaIds,
        _width, _height, 40, _maxTraceDepth, _onlyDirectLight);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);

    float time;
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("Time (GPU) : %f ms \n", time);

    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
}


void Render::phong() {
    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime, 0);

    dim3 tile = dim3(16, 16);
    dim3 block = dim3(_width + (tile.x - 1) / tile.x, _height + (tile.y - 1) / tile.y);

    phongKernel<<<block, tile>>>(
        _buffer, &_scene->_bvh, _scene->_bvh._leafNodes, _scene->_bvh._internalNodes,
        _scene->_pts, _scene->_indices, _scene->_geos, _scene->_materials,
        _emitAreas, _emitAreaSum, _emitAreaNum, _emitAreaIds,
        _width, _height, 40, _maxTraceDepth, _onlyDirectLight);

    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(startTime);
    cudaEventSynchronize(endTime);

    float time;
    cudaEventElapsedTime(&time, startTime, endTime);
    printf("Time (GPU) : %f ms \n", time);

    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
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
