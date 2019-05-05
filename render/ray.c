#include "render/ray.h"

ray ray_init(vec3 p, vec3 d)
{
	ray r;

	r.position = p;
	r.direction = d;
	return r;
}

vec3 ray_at_t(ray r, float t)
{
	return vec3_add(r.position, vec3_scale(r.direction, t)); 
}
