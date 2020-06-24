#ifndef __THREADED_RAYTRACER_H__
#define __THREADED_RAYTRACER_H__

#include "render/raytracer.h"

typedef struct {
	raytracer rt;
	int id;
	vec4 *c;
} threaded_args;

int threaded_render(const raytracer rt, void *cuda_rt);
void *threaded_render_work(void *args);
	
#endif
