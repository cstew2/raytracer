#ifndef __CPU_RAYTRACER_H__
#define __CPU_RAYTRACER_H__

#include "math/vector.h"
#include "world/colour.h"
#include "world/material.h"
#include "render/ray.h"
#include "render/raytracer.h"

vec3 reflection(const vec3 i, const vec3 n);
int refraction(const vec3 i, const vec3 n, const float a, vec3 *r);
int fresnel(const vec3 i, const vec3 n, const float a, vec3 *r);

int cpu_trace(const ray r, const raytracer rt, hit_info *hi);
colour cpu_cast_ray(const ray r, const raytracer rt);
int cpu_render(const raytracer rt);

#endif
