#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "math/vector.h"
#include "world/colour.h"
#include "render/ray.h"
#include "world/material.h"

typedef struct{
	vec3 position;
	float radius;
	colour c;
	material m;
}sphere;

sphere sphere_new(vec3 p, float r, colour c, material m);
int sphere_intersect(ray r, sphere s, float *t);

#endif
