#ifdef __NVCC__
extern "C" {
#endif	


#include "render/ray.hh"

__host__ __device__ ray ray_init(const vec3 p, const vec3 d)
{
	ray r;
	r.position = p;
	r.direction = d;
	return r;
}

__host__ __device__ vec3 ray_at_t(const ray r, const float t)
{
	return vec3_add(r.position, vec3_scale(r.direction, t)); 
}
	
#ifdef __NVCC__
}
#endif	
