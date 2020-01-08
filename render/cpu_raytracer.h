#ifndef __CPU_RAYTRACER_H__
#define __CPU_RAYTRACER_H__

#include "math/vector.h"
#include "world/colour.h"
#include "world/material.h"
#include "render/ray.h"
#include "render/raytracer.h"

typedef struct {
	colour hit_c;
	vec3 hit_n;
	vec3 hit_p;
	material hit_m;
}hit_info;

vec3 reflection(const vec3 i, const vec3 n);
vec3 refraction(const vec3 i, const vec3 n, const float a);
float fresnel(const vec3 i, const vec3 n, const float a);

int cpu_trace(const ray r, const raytracer rt, hit_info *hi);
colour cpu_cast_ray(const ray r, const raytracer rt);
int cpu_render(const raytracer rt);

#endif
