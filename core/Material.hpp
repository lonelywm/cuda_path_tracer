#pragma once
#include <pch.h>


struct Material {
    Vec3f ka; //Ambient
    Vec3f kd; //diffuse
    Vec3f ks; //specular
    Vec3f kt; //Transmittance
    Vec3f ke; //Emmission
    Vec3f kr; //reflectance == specular
    float shininess;
    float ior;
    float dissolve; // 1 == opaque; 0 == fully transparent

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

    __device__
    Material& operator += (const Material& m){
        ke += m.ke;
        ka += m.ka;
        ks += m.ks;
        kd += m.kd;
        kr += m.kr;
        kt += m.kt;
        ior += m.ior;
        shininess += m.shininess;
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
    m.shininess *= d;
    return m;
}
