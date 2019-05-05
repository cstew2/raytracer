#include <math.h>

#include "world/sphere.h"


sphere new_sphere(vec3 p, float r, colour c, material m)
{
	sphere s;
	s.position = p;
	s.radius = r;
	s.c = c;
	s.m = m;
	return s;
}

int sphere_intersect(ray r, sphere s, float *t)
{
	vec3 l = vec3_sub(r.position, s.position);
	float a = vec3_dot(r.direction, r.direction);
	float b = 2 * vec3_dot(l, r.direction);
	float c = vec3_dot(l, l) - s.radius;

	float discr = (b * b) - (4 * a * c);
	float t0;
	float t1;
	if(discr < 0.0) {
		return 0;
	}
	else if(discr == 0) {
		t0 = -0.5 * (b/a);
		t1 = t0;
	}

	else {
		float x = 0;
		if(b > 0) {
			x = -0.5 * (b + sqrt(discr));
		}
		else
		{
			x = -0.5 * (b - sqrt(discr));
		}
		t0 = x/a;
		t1 = c/x;
	}
	if(t0 > t1) {
		float q = t0;
		t0 = t1;
		t1 = q;
	}
	if(t0 < 0) {
		t0 = t1;
		if(t0 < 0) {
			return 0;
		}
	}
	*t = t0;
	return 1;
}
