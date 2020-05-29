#ifndef __PLANE_H__
#define __PLANE_H__

#include "math/vector.hh"
#include "render/ray.hh"

#include "world/material.h"
#include "main/cuda_check.h"

#ifdef __cplusplus
extern "C" {
#endif
	
typedef struct{
	vec3 position;
	vec3 normal;
	vec4 c;
	material m;
}plane;

plane plane_new(vec3 p, vec3 n, vec4 c, material m);
__host__ __device__ int plane_intersect(ray r, plane p, float *t);
__host__ __device__ void plane_hit(ray r, float t, plane p, hit_info *hi);

#ifdef __cplusplus
}
#endif
	
#endif
