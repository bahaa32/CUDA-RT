#ifndef CAMERA_H
#define CAMERA_H

#define BLOCKSIZE 640

#include "hittable_list.cuh"
#include "common.cuh"
#include "materials.cuh"
#include "materials.cu"

class camera {
friend __global__ void render_kernel(curandState* states, color* pixelsDst);
public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;  // How many random samples to take for each pixel
    int    max_depth = 10;   // Maximum number of ray bounces into scene
    double vfov = 90;  // Vertical view angle (field of view)

    point3 lookfrom = point3(0, 0, -1);  // Point camera is looking from
    point3 lookat = point3(0, 0, 0);   // Point camera is looking at
    vec3   vup = vec3(0, 1, 0);     // Camera-relative "up" direction
    
    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    int getHeight() {
        return image_height;
    }

    static void render(camera* cam, color** pixelsDst, void (*make_world)(hittable_list*));

protected:
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;        // Camera frame basis vectors
    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius

private:
    void initialize() {
        // Calculate the image height, and ensure that it's at least 1.
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Camera

        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ static color ray_color(curandState* state, ray& r, const int max_depth);
};

// I hate this ugly workaround
__constant__ char d_cam_c[sizeof(camera)];
__constant__ char d_world_c[sizeof(hittable_list)];

__global__ void init_curand_states(curandState* states, unsigned int imagesize, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < imagesize) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

void camera::render(camera* cam, color** pixelsDst, void (*make_world)(hittable_list*)) {
    cam->initialize();

    // Calculate space needed to store all pixel data.
    const int imagesize = cam->image_height * cam->image_width;
    const int datasize = sizeof(color) * imagesize;

    cudaErrorCheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, datasize + sizeof(hittable_list) * 5));

    // Allocate device memory for the image data and zero it out
    color* pixels;
    cudaErrorCheck(cudaMalloc((void**)&pixels, datasize));
    cudaErrorCheck(cudaMemsetAsync(pixels, 0, datasize));

    // Allocate device memory for the image data and zero it out
    curandState* states;
    cudaErrorCheck(cudaMalloc((void**)&states, imagesize*sizeof(curandState)));

    // Create the world on the device and move it to constant symbol
    hittable_list* world;
    cudaErrorCheck(cudaMalloc((void**)&world, sizeof(hittable_list)));
    make_world << <1, 1 >> > (world);
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpyToSymbol(d_world_c, world, sizeof(hittable_list), 0, cudaMemcpyDeviceToDevice));
    cudaFree(world);


    // Allocate memory for the camera on the device and copy it over
    cudaErrorCheck(cudaMemcpyToSymbol(d_cam_c, cam, sizeof(camera)));

    cudaErrorCheck(cudaDeviceSynchronize()); // Make sure all previous operations are done before launching the kernel
    auto start = std::chrono::high_resolution_clock::now();
    // Calculate grid size and launch the kernel!
    const int gridsize = (imagesize + BLOCKSIZE) / BLOCKSIZE;
    init_curand_states << <gridsize, BLOCKSIZE >> > (states, imagesize, 42);
    render_kernel << <gridsize, BLOCKSIZE >> > (states, pixels);
    std::clog << "Kernels launched. Waiting for task to complete..." << std::endl;

    // (tapping fingers...) wait for the GPU to complete the task
    cudaErrorCheck(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Mostly for debugging, check if anything went bad since we launched the kernels
    cudaErrorCheck(cudaGetLastError());

    std::clog << "Kernel execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Make space for the data on the host and copy it over
    (*pixelsDst) = (color*)malloc(datasize);
    cudaErrorCheck(cudaMemcpy(*pixelsDst, pixels, datasize, cudaMemcpyDeviceToHost));
    cudaFree(pixels);
}

__device__ __forceinline__ color camera::ray_color(curandState* state, ray& current_ray, const int max_depth) {
    const hittable_list* d_world = reinterpret_cast<const hittable_list*>(d_world_c);
    color result = color(1, 1, 1);
    hit_record rec;
    color attenuation;
    for (unsigned int i = 0; i < max_depth; i++) {
        if (d_world->hit(current_ray, interval(0.001, INFINITY), rec)) {
            if (scatter(state, rec, attenuation, current_ray)) {
                result = result * attenuation;
            }
            else {
                return color(0, 0, 0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(current_ray.direction());
            auto a = 0.5 * (unit_direction.y() + 1.0);
            return result * ((1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
        }
    }
    return result;
}

__global__ void render_kernel(curandState* global_states, color* pixelsDst) {
    const camera* d_cam = reinterpret_cast<const camera*>(d_cam_c);
    volatile unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_cam->image_width * d_cam->image_height) {
        unsigned int x = tid % d_cam->image_width;
        unsigned int y = tid / d_cam->image_width;
        __shared__ color pixel_centers[BLOCKSIZE];
        pixel_centers[threadIdx.x] = d_cam->pixel00_loc + (x * d_cam->pixel_delta_u) + (y * d_cam->pixel_delta_v);
        __shared__ color intermediate_pixels[BLOCKSIZE];
        intermediate_pixels[threadIdx.x] = color(0, 0, 0);
        curandState* state = &global_states[tid];
        ray r;
        for (int i = 0; i < d_cam->samples_per_pixel; i++) {
            auto px = -0.5 + random_double(state);
            auto py = -0.5 + random_double(state);
            auto pixel_sample = pixel_centers[threadIdx.x] + ((px * d_cam->pixel_delta_u) + (py * d_cam->pixel_delta_v));
            vec3 ray_origin;
            if (d_cam->defocus_angle <= 0) {
                ray_origin = d_cam->center;
            }
            else {
                auto p = random_in_unit_disk(state);
                ray_origin = d_cam->center + (p[0] * d_cam->defocus_disk_u) + (p[1] * d_cam->defocus_disk_v);
            }
            r.set(ray_origin, pixel_sample - ray_origin);
            intermediate_pixels[threadIdx.x] += camera::ray_color(state, r, d_cam->max_depth);
        }
        __syncthreads();
        pixelsDst[tid] = intermediate_pixels[threadIdx.x];
    }
}
#endif