#pragma once
#include <pch.h>
#include <curand_kernel.h>




struct Material {
    Vec3f ka; //Ambient
    Vec3f kd; //diffuse
    Vec3f ks; //specular
    Vec3f kt; //Transmittance
    Vec3f ke; //Emmission
    Vec3f kr; //reflectance == specular

    float ior;
    float dissolve; // 1 == opaque; 0 == fully transparent

    uint beginIndex;  // device
    uint endIndex;    // device

    bool _refl;								  // specular reflector?
    bool _trans;							  // specular transmitter?
    bool _spec;								  // any kind of specular?
    bool _both;								  // reflection and transmission

    __host__ __device__
    void setBools() {
        _refl  = isZero(kr);
        _trans = isZero(kt);
        _spec  = _refl || isZero(ks);
        _both  = _refl && _trans;
    }

    __host__ __device__
    bool Refl() const {
        return _refl;
    }
    __host__ __device__
    bool Trans() const {
        return _trans;
    }

    __device__
    Vec3f shade() {
        Vec3f I = kd;
        return I;
    }

    __host__ __device__
    bool hasEmission() {
        if (ke.norm2() > EPSILON) return true;
        return false;
    }

    __device__
    Vec3f toWorld(CVec3f& a, CVec3f& N) {
        Vec3f B, C;
        if (fabs(N.x) > fabs(N.y)) {
            float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
            C = Vec3f(N.z * invLen, 0.0f, -N.x * invLen);
        } else {
            float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
            C = Vec3f(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = C.cross(N);
        return a.x * B + a.y * C + a.z * N;
    }


    __host__ __device__ __inline__
    Vec3f eval(CVec3f& wi, CVec3f& wo, CVec3f& nor) {
        // diffuse
        // calculate the contribution of diffuse   model
        float cosalpha = nor.dot(wo);
        if (cosalpha > 0.0f) {
            Vec3f diffuse = kd / PI;
            return diffuse;
        }
        return Vec3f();
    }

    __device__
    Vec3f sample(CVec3f& Nor, curandState& randState) {
        float x1 = curand_uniform(&randState);
        float x2 = curand_uniform(&randState);

        float z = std::fabs(1.0f - 2.0f * x1);
        float r = std::sqrt(1.0f - z * z);
        float phi = 2 * PI * x2;
        Vec3f localRay(r * cos(phi), r * sin(phi), z);
        return toWorld(localRay, Nor).normalize();
    }

    __device__
    float Material::pdf(const Vec3f& wo, const Vec3f& N) {
        if (wo.dot(N) > 0.0f)
            return 0.5f / PI;
        else
            return 0.0f;
    }

    __device__
    Material& operator += (const Material& m){
        ke += m.ke;
        ka += m.ka;
        ks += m.ks;
        kd += m.kd;
        kr += m.kr;
        kt += m.kt;
        ior += m.ior;
        setBools();
        return *this;
    }

    friend __device__ __inline__
    Material operator*(float d, Material m);

};

__device__ __inline__
Material operator*(float d, Material m){
    m.ke *= d;
    m.ka *= d;
    m.ks *= d;
    m.kd *= d;
    m.kr *= d;
    m.kt *= d;
    m.ior *= d;
    return m;
}
