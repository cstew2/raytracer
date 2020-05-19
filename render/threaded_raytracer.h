#ifndef __THREADED_RAYTRACER_H__
#define __THREADED_RAYTRACER_H__

#include "render/raytracer.h"

typedef struct {
	raytracer rt;
	int id;
	colour *c;
} threaded_args;

int threaded_render(const raytracer rt);
void *threaded_render_work(void *args);
	
#endif
