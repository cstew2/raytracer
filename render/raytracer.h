#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "render/camera.h"
#include "world/scene.h"
#include "render/canvas.h"
#include "main/config.h"

typedef struct {
	config config;
	camera camera;
	scene *objects;
	canvas canvas;
}raytracer;

raytracer raytracer_init(config c, camera cam, scene *objs);
void raytracer_term(raytracer rt);
raytracer raytracer_test(config c);

#endif
