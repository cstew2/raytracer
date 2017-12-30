#include <stdbool.h>

#include "raytracer.h"


bool open;
canvas c1;
canvas c2;


canvas new_canvas(int width, int height, colour *c)
{
	canvas cv;
	cv.width = width;
	cv.height = height;
	cv.screen = c;
	return cv;
}

ray new_ray(vec3 p, vec3 d)
{
	ray r;
	r.pos = p;
	r.dir = d;
	return r;
}

void update(void)
{
	if(open) {

		open = false;
	}
	else {

		open = true;
	}
}
