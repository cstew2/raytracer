#ifndef __CUDA_RAYTRACER_CU__
#define __CUDA_RAYTRACER_CU__

#ifdef __cplusplus
extern "C" {
#endif
#include "math/vector.h"
#include "world/colour.h"
#include "world/material.h"
#include "render/ray.h"
#include "render/raytracer.h"
#ifdef __cplusplus	
}
#endif

int cuda_trace(const ray r, const raytracer rt, hit_info *hi);
colour cuda_cast_ray(const ray r, const raytracer rt);
int cuda_render(const raytracer rt);

#endif
