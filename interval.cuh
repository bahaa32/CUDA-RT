#ifndef INTERVAL_H
#define INTERVAL_H

#include "math.h"

class interval {
public:
    double min, max;

    __device__ interval() : min(+INFINITY), max(-INFINITY) {} // Default interval is empty

    __device__ interval(double _min, double _max) : min(_min), max(_max) {}

    __device__ inline bool contains(double x) const {
        return min <= x && x <= max;
    }

    __device__ inline bool surrounds(double x) const {
        return min < x && x < max;
    }

    __device__ inline double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

const static interval empty(+INFINITY, -INFINITY);
const static interval universe(-INFINITY, +INFINITY);

#endif