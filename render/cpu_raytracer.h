#ifndef __CPU_RAYTRACER_H__
#define __CPU_RAYTRACER_H__

#include "math/vector.h"
#include "world/colour.h"
#include "world/material.h"
#include "render/ray.h"
#include "render/raytracer.h"

float ambient(const float a, const float k_a);
float diffuse();

vec3 specular_reflection(const vec3 i, const vec3 n);
vec3 refraction(const vec3 i, const vec3 n, const float a);
float fresnel(const vec3 i, const vec3 n, const float a, const float b);

int cpu_trace(const ray r, const raytracer rt, hit_info *hi);
colour cpu_cast_ray(const ray r, const raytracer rt);
int cpu_render(const raytracer rt);

#endif
