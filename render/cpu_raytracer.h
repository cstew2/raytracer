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

int cpu_trace(ray r, hit_info *hi);
colour cpu_cast_ray(ray r);
int cpu_render(raytracer rt);

#endif
