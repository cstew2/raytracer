#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "math/vector.h"
#include "world/colour.h"

typedef struct {
	colour c;
	float intensity;
	vec3 position;
	vec3 direction;
}light;

light light_new(colour c, float intensity, vec3 position, vec3 direction);

#endif
