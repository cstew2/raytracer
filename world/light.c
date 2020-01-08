#include "world/light.h"

light light_new(colour c, float intensity, vec3 position, vec3 direction)
{
	light l;
	l.c = c;
	l.intensity = intensity;
	l.position = position;
	l.direction = direction;
	return l;
}
