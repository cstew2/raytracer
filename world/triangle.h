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
	material h;
}triangle;

triangle new_triangle(vec3 v1, vec3 v2, vec3 v3, colour c);
int triangle_intersect(ray r, triangle t, vec3 *intersect);
	
#endif
