#include "world/light.h"

light light_new(vec4 c, float intensity, float spread, vec3 position, vec3 direction)
{
	light l;
	l.c = c;
	l.intensity = intensity;
	l.spread = spread;
	l.position = position;
	l.direction = vec3_normalize(direction);
	return l;
}
