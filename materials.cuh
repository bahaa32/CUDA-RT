#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.cuh"

class hit_record;

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
};

// This struct contains all the information needed to describe any material
struct material {
    // All
    MaterialType type;
    color albedo;

    // Metal
    double roughness;

    // Dielectric
    float refractive_index;
};

__device__ bool scatter(curandState* state, const hit_record& rec, color& attenuation, ray& scattered);

#endif