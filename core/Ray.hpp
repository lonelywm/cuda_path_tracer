#pragma once
#include <pch.h>

class Ray {
public:
    Vec3f Pos;
    Vec3f Dir;

public:
    __host__ __device__
    Ray(const Vec3f &p, const Vec3f &d): Pos(p), Dir(d) {}

    __host__ __device__
    Ray(const Ray& other): Pos(other.Pos), Dir(other.Dir) {}

    __host__ __device__
    ~Ray() {}

    __host__ __device__
    Ray& operator =( const Ray& other ) 
    { Pos = other.Pos; Dir = other.Dir; return *this; }

    __host__ __device__
    Vec3f at( float t ) const
    { return Pos + (t*Dir); }

    __host__ __device__
    Vec3f getPosition() const { return Pos; }

    __host__ __device__
    Vec3f getDirection() const { return Dir; }
};
