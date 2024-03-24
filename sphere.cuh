#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "materials.cuh"
#include "vec3.cuh"

class sphere : public hittable {
public:
    __device__ sphere(point3 _center, double _radius, material _material) : center(_center), radius(_radius), mat(_material) {}

    __device__ bool hit(const ray & r, interval ray_t, hit_record & rec) const override {
    vec3 oc = r.origin() - center;
    auto half_b = dot(oc, r.direction());
    auto a = r.direction().length_squared();
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    auto inv_a = 1.0 / a; // Calculate the inverse early, reduce the need for division
    auto root = (-half_b - sqrtd) * inv_a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) * inv_a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = mat;

    return true;
}

    __device__ int get_size() const override {
        return sizeof(sphere);
    }

private:
    material mat;
    point3 center;
    double radius;
};

#endif