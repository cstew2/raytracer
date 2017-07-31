#include "camera.h"

camera new_camera(v3 p, v3 l, v3 u)
{
	camera c;
	c.position = p;
	c.lookat = l;
	c.up = u;
	return c;
}
