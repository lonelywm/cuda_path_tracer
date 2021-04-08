#pragma once
#include <pch.h>

struct Point {
    Vec3f Pos;
    Vec3f Nor;
    Vec2f Tex;
    uint MIds[MAX_MATERIAL_COUNT];  // used in device; MaterialIds
    uint MNum;                      // Material Count
};