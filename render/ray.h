#ifndef __RAY_H__
#define __RAY_H__

#include "math/vector.h"

typedef struct {
	vec3 position;
	vec3 direction;
}ray;

ray ray_init(vec3 p, vec3 d);
vec3 ray_at_t(ray r, float t);

#endif
