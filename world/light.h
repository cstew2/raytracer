#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "math/vector.hh"

typedef struct {
	vec4 c;
	float intensity;
	float spread;
	vec3 position;
	vec3 direction;
}light;

light light_new(vec4 c, float intensity, float spread, vec3 position, vec3 direction);

#endif
