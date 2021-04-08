#pragma once
#include <pch.h>
#include "Scene.h"


class Render {
public:
    
    void rend(Scene* scene, int width, int height, int spp);
    void output();

    __device__
    void shade(const Intersection& isect, const Vec3f& indir);

    __device__
    void sampleLight(Vec3f* poss, uint* indices, int numTri, Intersection &pos, float &pdf, curandState_t& randState) const;

public:
    Vec3f* _buffer;
    int _width;
    int _height;
    std::vector<Vec3f> framebuffer;
};