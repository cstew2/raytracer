#include "camera.h"

camera new_camera(vect3 p, vect3 l, vect3 u)
{
	camera c;
	c.position = p;
	c.lookat = l;
	c.up = u;
	return c;
}
