#ifndef __COMMON_RAYTRACER_H__
#define __COMMON_RAYTRACER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "main/cuda_check.h"
#include "math/vector.hh"

__host__ __device__ vec4 ambient(const vec4 a, const float i);
__host__ __device__ vec4 diffuse(const vec3 on, const vec4 oc, const float ck,
				 const vec3 lv, const float li, const vec4 lc);
__host__ __device__ vec4 specular_reflection(const vec3 on, const vec4 oc, const float ck,
					     const vec3 lv, const float li, const vec4 lc,
					     const vec3 dv, const float p);
__host__ __device__ vec4 reflection(const vec3 i, const vec3 n);
__host__ __device__ vec4 refraction(const vec3 i, const vec3 n, const float a);
__host__ __device__ vec4 fresnel(const vec3 i, const vec3 n, const float a, const float b);

__host__ __device__ float shadow();

#ifdef __cplusplus
}
#endif

#endif
