#ifndef __PLANE_H__
#define __PLANE_H__

#include "math/vector.h"
#include "world/colour.h"
#include "render/ray.h"
#include "world/material.h"

typedef struct{
	vec3 position;
	vec3 normal;
	colour c;
	material m;
}plane;

plane new_plane(vec3 p, vec3 n, colour c, material m);
int plane_intersect(ray r, plane p, float *t);
	
#endif
