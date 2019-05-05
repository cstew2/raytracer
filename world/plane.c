#include "world/plane.h"

plane new_plane(vec3 p, vec3 n, colour c, material m)
{
	plane pl;
	pl.position = p;
	pl.normal = n;
	pl.c = c;
	pl.m = m;
	return pl;
}

int plane_intersect(ray r, plane p, float *t)
{	
	float d = vec3_dot(p.normal, r.direction);
	if(d > 1e-6) {
		vec3 x = vec3_sub(p.position, r.position);
		float n = vec3_dot(x, p.normal);
		*t = n / d;
		return (*t >= 0);
	}
	return 0;
}
