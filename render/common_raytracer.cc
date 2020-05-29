#include <math.h>

#ifdef __NVCC__
extern "C" {
#endif	

#include "math/math.h"
	
#include "render/common_raytracer.hh"



__host__ __device__ vec4 ambient(const vec4 a, const float i)
{
	return vec4_scale(a, i);
}

__host__ __device__ vec4 diffuse(const vec3 on, const vec4 oc, const float ck,
	       const vec3 lv, const float li, const vec4 lc)
{
	//oc * ((on.lv) * k * li)
	float i = max(0.0, vec3_dot(on, lv));
	return vec4_scale3(oc, li*i*ck);
}

__host__ __device__ vec4 specular_reflection(const vec3 on, const vec4 oc, const float ck,
			   const vec3 lv, const float li, const vec4 lc,
			   const vec3 dv, const float p)
{
	vec3 h = vec3_normalize(vec3_add(lv, dv));
	float i = powf(vec3_dot(h, on), p);
	return vec4_new(i, i, i, 1.0);
}

__host__ __device__ vec4 reflection(const vec3 i, const vec3 n)
{
	//r=i-n(2(n.i))
	//vec3_sub(i, vec3_scale(n, 2*vec3_dot(i, n)));
}

__host__ __device__ vec4 refraction(const vec3 i, const vec3 n, const float a)
{
	//
}

__host__ __device__ vec4 fresnel(const vec3 i, const vec3 n, const float a, const float b)
{
	//
}

#ifdef __NVCC__
}
#endif	
