#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include "math/vector.h"
#include "voxel/dag.h"
#include "world/colour.h"
#include "world/camera.h"

typedef struct {
	int width;
	int height;
	colour *screen;
}canvas;

typedef struct {
	vec3 pos;
	vec3 dir;
}ray;

canvas new_canvas(int width, int height, colour *c);
ray new_ray(vec3 p, vec3 d);
void update(void);

#endif
