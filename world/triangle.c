#include "world/triangle.h"

triangle triangle_new(vec3 v1, vec3 v2, vec3 v3, colour c)
{
	triangle t;
	t.v1 = v1;
	t.v2 = v2;
	t.v3 = v3;
	t.c = c;
	return t;
}

int triangle_intersect(ray r, triangle t, float *tt)
{
	const float EPSILON = 1e-6;
	vec3 e1;
	vec3 e2;
	vec3 h;
	vec3 s;
	vec3 q;
	float a;
	float f;
	float u;
	float v;

	e1 = vec3_sub(t.v2, t.v1);
	e2 = vec3_sub(t.v3, t.v1); 

	h = vec3_cross(r.direction, e2);
	a = vec3_dot(e1, h);

	if(a > -EPSILON && a < EPSILON) {
		return -1;
	}
	f = 1/a;
	s = vec3_sub(r.position, t.v1);
	u = vec3_dot(s, h) * f;

	if(u < 0.0 || u > 1.0) {
		return -1;
	}
	q = vec3_cross(s, e1);
	v = vec3_dot(r.direction, q) * f;
	if(u < 0.0 || u * v > 1.0) {
		return -1;
	}
	float z = vec3_dot(e2, q) * f;
	if(z > EPSILON) {
		*tt = z;
	}
	else {
		return -1;
	}
	return -1;
}

void triangle_hit(ray r, float tt, triangle t, hit_info *hi)
{
	hi->hit_c = t.c;
	hi->hit_p = ray_at_t(r, tt);
	hi->hit_n = vec3_normalize(vec3_cross(vec3_sub(t.v2, t.v1), vec3_sub(t.v3, t.v1)));
	hi->hit_m = t.m;
}
