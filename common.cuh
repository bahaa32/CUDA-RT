#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include "cuda_runtime.h"
#include <curand_kernel.h>

//#define cudaErrorCheck(ans) { ans; } // disables cudaErrorCheck
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__host__ inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

// Constants
__device__ const double pi = 3.1415926535897932385;

// Utility Functions

__device__ __host__ __forceinline__ double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ __forceinline__ double random_double(curandState* state) {
    // Returns a random double in (0,1].
    return curand_uniform_double(state);
}

__device__ __forceinline__ double random_double(curandState* state, double min, double max) {
    // Returns a random double in (min,max].
    return min + (max - min) * random_double(state);
}

// Common Headers

#include "ray.cuh"
#include "vec3.cuh"
#include "color.cuh"

#endif