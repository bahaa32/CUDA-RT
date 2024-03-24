#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

class ray {
public:
    __device__ __host__ ray() {}

    __device__ __host__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ __host__ point3 origin() const { return orig; }
    __device__ __host__ vec3 direction() const { return dir; }

    __device__ __host__ point3 at(double t) const {
        return orig + t * dir;
    }

    __device__ __host__ inline ray copy() const {
        return ray(orig, dir);
    }

    __device__ __host__ inline void set(const point3& origin, const vec3& direction) {
        orig = origin;
        dir = direction;
    }


private:
    point3 orig;
    vec3 dir;
};

#endif