#ifndef __RAY_H__
#define __RAY_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "math/vector.hh"
#include "world/material.h"
#include "main/cuda_check.h"

typedef struct {
	vec3 position;
	vec3 direction;
}ray;

typedef struct {
	vec4 hit_c;
	vec3 hit_n;
	vec3 hit_p;
	material hit_m;
}hit_info;



__host__ __device__ ray ray_init(const vec3 p, const vec3 d);
__host__ __device__ vec3 ray_at_t(const ray r, const float t);

#ifdef __cplusplus
}
#endif
	
#endif
