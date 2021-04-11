#pragma once
#include <pch.h>
#include "Scene.h"

class Render {
public:
    
    void rend(Scene* scene, int width, int height, int spp, int maxTraceDepth = 4);
    void output();

    __device__
    void shade(const Intersection& isect, const Vec3f& indir);

public:
    float* _emitAreas;        // device 发光面积
    float  _emitAreaSum = 0;  // 发光面积总数
    uint   _emitAreaNum = 0;  // 发光物体数
    uint*  _emitAreaIds;      // 发光物体的排序 （面积大的在前面，不发光的面积=0）

public:
    Vec3f* _buffer;
    int _width;
    int _height;
    std::vector<Vec3f> framebuffer;
};