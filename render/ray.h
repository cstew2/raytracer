#ifndef __RAY_H__
#define __RAY_H__

#include "math/vector.h"
#include "world/material.h"
#include "world/colour.h"

typedef struct {
	vec3 position;
	vec3 direction;
}ray;

typedef struct {
	colour hit_c;
	vec3 hit_n;
	vec3 hit_p;
	material hit_m;
}hit_info;

ray ray_init(vec3 p, vec3 d);
vec3 ray_at_t(ray r, float t);

#endif
