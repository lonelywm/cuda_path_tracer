#pragma once
#include <vector>
#include <string>
#include <array>
#include <matrix.h>

#define EPSILON 1e-7

#ifdef USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif

#define HD __host__ __device__
#define HO __host__
#define DE __device__
using uint = unsigned int;

// #define printf(...) 

// using Vector2i     = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
// using Vector3i     = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;
// using Vector2r     = Eigen::Matrix<Real, 2, 1, Eigen::DontAlign>;
// using Vector3r     = Eigen::Matrix<Real, 3, 1, Eigen::DontAlign>;
// using Vector4r     = Eigen::Matrix<Real, 4, 1, Eigen::DontAlign>;
// using Matrix2r     = Eigen::Matrix<Real, 2, 2, Eigen::DontAlign>;
// using Matrix3r     = Eigen::Matrix<Real, 3, 3, Eigen::DontAlign>;
// using Matrix4r     = Eigen::Matrix<Real, 4, 4, Eigen::DontAlign>;
// using CVector2r = const Vector2r;
// using CVec3f = const Vector3r;
// using CVector4r = const Vector4r;
// using CVec3f = const Vector3r;
// using CMatrix2r = const Matrix2r;
// using CMatrix3r = const Matrix3r;
// using CMatrix4r = const Matrix4r;

// using AABB2        = Eigen::AlignedBox<Real, 2>;
// using AABB3        = Eigen::AlignedBox<Real, 3>;
// using AngleAxisr   = Eigen::AngleAxis<Real>;
// using Quaternionr  = Eigen::Quaternion<Real, Eigen::DontAlign>;
using CVec3f = const Vec3f;
using Vec2f  = Vec3f;
using CVec2f = const Vec3f;

template<typename T>
using Vector = std::vector<T>;
using String = std::string;

template<typename T>
using CVector = const Vector<T>;

template<typename P, size_t Q>
using Array = std::array<P, Q>;

struct Transform3 {
    Vec3f Position, Rotation, Scale;
    Transform3() { Scale = Vec3f(1.f, 1.f, 1.f); }
};
