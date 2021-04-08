#pragma once
#include <pch.h>
#include "Mesh.hpp"
#include "Texture.hpp"
#include "BVH.h"

class Actor {
public:
    Transform3      Transform;
    Vec3f           Velocity;
    Vector<Mesh*>   Meshes;

private:
    std::string  _path;  // local file path
};