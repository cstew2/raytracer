#ifndef __CUDA_RAYTRACER_CU__
#define __CUDA_RAYTRACER_CU__

#include "render/camera.hh"
#include "world/sphere.hh"
#include "world/plane.hh"
#include "render/ray.hh"
#include "math/vector.hh"
#include "render/common_raytracer.hh"

#ifdef __NVCC__
extern "C" {
#endif	
#include "world/material.h"
#include "render/raytracer.h"
#ifdef __NVCC__
}
#endif

__device__ int cuda_trace(const ray r, const raytracer *rt, hit_info *hi);
__device__ vec4 cuda_cast_ray(const ray r, const raytracer *rt);
__global__ void cuda_render_kernel(raytracer *rt);

#ifdef __NVCC__
extern "C" {
#endif	
int cuda_render(raytracer rt);
#ifdef __NVCC__
}
#endif

#endif
