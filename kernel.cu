
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>

#include "common.cuh"

#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "materials.cuh"


#define FILENAME "image.ppm"

inline double linear_to_gamma(double linear_component)
{
    return sqrt(linear_component);
}

void print_p3(point3* pixels, int image_width, int image_height, int samples_per_pixel) {
    std::clog << "Started writing..." << std::endl;
    std::ofstream file(FILENAME);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    file << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    const int imagesize = image_width * image_height;
    for (int i = 0; i < imagesize; i++) {
        point3 pixel = pixels[i];
        int ir = linear_to_gamma(pixel.x() / samples_per_pixel) * 255.999;
        int ig = linear_to_gamma(pixel.y() / samples_per_pixel) * 255.999;
        int ib = linear_to_gamma(pixel.z() / samples_per_pixel) * 255.999;
        file << ir << ' ' << ig << ' ' << ib << '\n';
    }
    file.close();
    std::clog << "Done (Wrote " << image_height << " scanlines or " << image_width * image_height << " pixels)." << std::endl;
}

// We use a function to build the world on GPU here since virtual function
//  pointers continue to point to the host's memory, so this workaround
//  is necessary such that we define the objects on device.
__global__ void make_world(hittable_list* worldPtr) {
    curandState state;
    curand_init(42, 0, 0, &state);
    auto world = new hittable_list();
    // material material_ground = material{ LAMBERTIAN, {0.8, 0.8, 0.0} };
    // material material_center = material{ DIELECTRIC, {0.7, 0.3, 0.3}, 0, 1.5 };
    // material material_left = material{ METAL, {0.8, 0.8, 0.8}, 1.0 };
    // material material_right = material{ LAMBERTIAN, {0.8, 0.4, 0.2} };
    // world->add(&sphere(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    // world->add(&sphere(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    // world->add(&sphere(point3(-1.0, 0.0, -1.0), 0.4, material_left));
    // world->add(&sphere(point3(0.0, 0.0, -1.0), 0.5, material_center));
    // world->add(&sphere(point3(1.0, 0.0, -1.0), 0.5, material_right));

    material ground_material = material{LAMBERTIAN, {0.5, 0.5, 0.5}};
    world->add(&sphere(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double(&state);
            point3 center(a + 0.9*random_double(&state), 0.2, b + 0.9*random_double(&state));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material sphere_material;

                if (choose_mat < 0.7) {
                    // diffuse
                    auto albedo = color::random(&state) * color::random(&state);
                    sphere_material = material {LAMBERTIAN, albedo};
                    world->add(&sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.8) {
                    // metal
                    auto albedo = color::random(&state, 0.5, 1);
                    auto fuzz = random_double(&state, 0, 0.5);
                    sphere_material = material{ METAL, albedo, fuzz };
                    world->add(&sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.9){
                    // glass
                    auto albedo = color::random(&state);
                    sphere_material = material{DIELECTRIC, albedo, 0, 1.5};
                    world->add(&sphere(center, 0.2, sphere_material));
                } else {
                    sphere_material = material{ DIELECTRIC, color(1,1,1), 0, 1.5 };
                    world->add(&sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = material{DIELECTRIC, color(1,1,1), 0, 1.5};
    world->add(&sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = material{ LAMBERTIAN, color(0.4, 0.2, 0.1) };
    world->add(&sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = material{ METAL, color(0.7, 0.6, 0.5), 0.0 };
    world->add(&sphere(point3(4, 1, 0), 1.0, material3));
               
    memcpy(worldPtr, world, sizeof(hittable_list));
}

int main() {
    std::clog << "Initializing CUDA subsystem..." << std::endl;
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // Image
    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 15360;
    cam.samples_per_pixel = 1;
    cam.max_depth = 10;

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    color* pixels;
    cam.render(&cam, &pixels, make_world);

    print_p3(pixels, cam.image_width, cam.getHeight() , cam.samples_per_pixel);

    free(pixels);
    // Destroying our CUDA context will free all the memory we allocated and
    //  is useful for profilers.
    cudaDeviceReset();
}
