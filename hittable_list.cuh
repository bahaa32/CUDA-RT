#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "common.cuh"
#include "hittable.cuh"
#include "sphere.cuh"

#include <memory>
#include <vector>

#define MAX_OBJECTS 512

// Hittable list data lives in CUDA MEMORY! It must be created on device.
class hittable_list : public hittable {
public:
    hittable* objects[MAX_OBJECTS];
    int len = 0;

    __device__ void add(hittable* object) {
        hittable* d_object = (hittable*)malloc(object->get_size());
        if (d_object == nullptr) {
            printf("Failed to allocate memory for hittable object\n");
            return;
        }
        memcpy(d_object, object, object->get_size());
        objects[len++] = d_object;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < len; i++) {
            auto object = objects[i];
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ void free() {
        for (int i = 0; i < len; i++) {
            cudaErrorCheck(cudaFree(objects[i]));
        }
    }

    __device__ int get_size() const override {
        return sizeof(hittable_list);
    }
};

#endif