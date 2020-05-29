#ifndef __CPU_RAYTRACER_H__
#define __CPU_RAYTRACER_H__

#include "math/vector.hh"
#include "render/ray.hh"

#include "world/material.h"
#include "render/raytracer.h"
#include "main/cuda_check.h"

int cpu_trace(const ray r, const raytracer rt, bool shadow, hit_info *hi);
vec4 cpu_cast_ray(const ray r, const raytracer rt);
int cpu_render(const raytracer rt);

#endif
