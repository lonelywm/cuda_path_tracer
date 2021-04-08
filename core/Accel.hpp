#pragma once
#include "BBox.hpp"


struct Accel {
    virtual const BoundingBox& getBBox() = 0;
};