#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include "math/vector.h"
#include "world/colour.h"
#include "render/ray.h"
#include "world/material.h"

typedef struct {
	vec3 v1;
	vec3 v2;
	vec3 v3;
	colour c;
	material m;
}triangle;

triangle triangle_new(vec3 v1, vec3 v2, vec3 v3, colour c);
int triangle_intersect(ray r, triangle t, float *tt);
void triangle_hit(ray r, float tt, triangle t, hit_info *hi);
	
#endif
