#include <math.h>

#ifdef __NVCC__
extern "C" {
#endif	

#include "world/plane.hh"

plane plane_new(vec3 p, vec3 n, vec4 c, material m)
{
	plane pl;
	pl.position = p;
	pl.normal = n;
	pl.c = c;
	pl.m = m;
	return pl;
}



__host__ __device__ int plane_intersect(ray r, plane p, float *t)
{	
	float d = vec3_dot(p.normal, r.direction);
	if(fabs(d) > 1e-6) {
	        *t = vec3_dot(vec3_sub(p.position, r.position), p.normal)/d;
		return (*t >= 0);
	}
	return 0;
}

__host__ __device__ void plane_hit(ray r, float t, plane p, hit_info *hi)
{
	hi->hit_c = p.c;
	hi->hit_p = ray_at_t(r, t);
	hi->hit_n = p.normal;
	hi->hit_m = p.m;
}

#ifdef __NVCC__
}
#endif	
