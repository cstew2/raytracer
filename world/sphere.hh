#ifndef __SPHERE_H__
#define __SPHERE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "math/vector.hh"
#include "render/ray.hh"

#include "world/material.h"
#include "main/cuda_check.h"

typedef struct{
	vec3 position;
	float radius;
	vec4 c;
	material m;
}sphere;

sphere sphere_new(vec3 p, float r, vec4 c, material m);
__host__ __device__ int sphere_intersect(ray r, sphere s, float *t);
__host__ __device__ void sphere_hit(ray r, float t, sphere s, hit_info *hi);

#ifdef __cplusplus
}
#endif
	
#endif
