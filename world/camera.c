#include "camera.h"

camera new_camera(vec3 p, vec3 l, vec3 u)
{
	camera c;
	c.position = p;
	c.lookat = l;
	c.up = u;
	return c;
}
