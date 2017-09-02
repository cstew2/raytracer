#include "raytracer.h"

ray new_ray(vec3 p, vec3 d)
{
	ray r;
	r.pos = p;
	r.dir = d;
	return r;
}
