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

vec3 reflection(vec3 incident, vec3 normal);
int trace(ray r, raytracer rt, hit_info *hi);
colour cast_ray(ray r, raytracer rt);

int render(raytracer rt);

#endif
