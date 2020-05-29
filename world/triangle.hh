#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "math/vector.hh"
#include "render/ray.hh"
#include "world/material.h"

typedef struct {
	vec3 v1;
	vec3 v2;
	vec3 v3;
	vec4 c;
	material m;
}triangle;

triangle triangle_new(vec3 v1, vec3 v2, vec3 v3, vec4 c);
__host__ __device__ int triangle_intersect(ray r, triangle t, float *tt);
__host__ __device__ void triangle_hit(ray r, float tt, triangle t, hit_info *hi);

#ifdef __cplusplus
}
#endif
	
#endif
