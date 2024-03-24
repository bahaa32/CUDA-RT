#ifndef VEC3_H
#define VEC3_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_atomic_functions.h>
#include <iostream>

class vec3 {
public:
    double e[3];

    __host__ __device__ vec3() : e{ 0,0,0 } {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __host__ __device__ inline double x() const { return e[0]; }
    __host__ __device__ inline double y() const { return e[1]; }
    __host__ __device__ inline double z() const { return e[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline double operator[](int i) const { return e[i]; }
    __host__ __device__ inline double& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator/=(double t) {
        float inv_t = 1.0f / t; // only approximation since this isn't important
        return *this *= inv_t;
    }

    __device__ __host__ inline double length() const {
        return sqrt(length_squared());
    }

    __device__ __host__ inline double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ inline static vec3 random(curandState* state) {
        return vec3(random_double(state), random_double(state), random_double(state));
    }


    __device__ inline static vec3 random(curandState* state, double min, double max) {
        return vec3(random_double(state, min, max), random_double(state, min, max), random_double(state, min, max));
    }

    __device__ inline bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        constexpr float epsilon = 1e-8;
        return (fabsf(e[0]) < epsilon) && (fabsf(e[1]) < epsilon) && (fabsf(e[2]) < epsilon);
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t) {
    return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_sphere(curandState* state) {
    while (true) {
        auto p = vec3::random(state, -1, 1);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState* state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ inline vec3 random_on_hemisphere(curandState* state, const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline vec3 random_in_unit_disk(curandState* state) {
    while (true) {
        auto p = vec3(random_double(state, -1, 1), random_double(state, -1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

#endif