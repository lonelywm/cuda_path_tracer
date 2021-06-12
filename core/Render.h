#pragma once
#include <pch.h>
#include "Scene.h"

class Render {
public:
    void render();
    void phong();
    void init(Scene* scene, int width, int height, int spp, int maxTraceDepth = 4, bool onlyDirectLight = false);
    void output();

public:
    float* _emitAreas;        // device 发光面积
    float  _emitAreaSum = 0;  // 发光面积总数
    uint   _emitAreaNum = 0;  // 发光物体数
    uint*  _emitAreaIds;      // 发光物体的排序 （面积大的在前面，不发光的面积=0）

public:
    Vec3f* _buffer;
    int _width;
    int _height;
    Scene* _scene;
    int _spp = 4;
    int _maxTraceDepth = 4;
    bool _onlyDirectLight = false;
    std::vector<Vec3f> framebuffer;
};