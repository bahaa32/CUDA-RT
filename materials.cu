#pragma once
#include "materials.cuh"
#include "hittable.cuh"

__device__ __forceinline__ bool lambertian_scatter(curandState* state, const hit_record& rec, color& attenuation, ray& scattered) {
    vec3 scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
    {
        scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = rec.mat.albedo;
    return true;
}

__device__ __forceinline__ bool metal_scatter(curandState* state, const hit_record& rec, color& attenuation, ray& scattered) {
    scattered = ray(rec.p, reflect(unit_vector(scattered.direction()), rec.normal) + rec.mat.roughness * random_unit_vector(state));
    attenuation = rec.mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

// Schlick's approximation for reflectance.
__device__ __forceinline__ double reflectance(double cosine, double ref_idx) {
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ __forceinline__ bool dielectric_scatter(curandState* state, const hit_record& rec, color& attenuation, ray& scattered) {
    const double refraction_ratio = rec.front_face ? (1.0 / rec.mat.refractive_index) : rec.mat.refractive_index;

    const vec3 unit_direction = unit_vector(scattered.direction());
    const double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    scattered = ray(rec.p, (refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > random_double(state)) ? reflect(unit_direction, rec.normal) : refract(unit_direction, rec.normal, refraction_ratio));
    attenuation = rec.mat.albedo;
    return true;
}

__device__ __forceinline__ bool scatter(curandState* state, const hit_record& rec, color& attenuation, ray& scattered) {
    switch (rec.mat.type) {
        case LAMBERTIAN: {
            return lambertian_scatter(state, rec, attenuation, scattered);
        }
        case METAL: {
            return metal_scatter(state, rec, attenuation, scattered);
        }
        case DIELECTRIC: {
            return dielectric_scatter(state, rec, attenuation, scattered);
        }
    }
}