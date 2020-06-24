#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "render/camera.hh"
#include "world/scene.h"
#include "render/canvas.hh"
#include "main/config.h"

typedef struct {
	bool forward;
	bool left;
	bool right;
	bool back;
	bool up;
	bool down;

	double last_x;
	double last_y;
	float yaw;
	float pitch;
	float speed;
	float sensitivity;
}state;

typedef struct {
	config config;
	camera camera;
	scene *objects;
	canvas canvas;
	state state;
}raytracer;

raytracer raytracer_init(config c, camera cam, scene *objs);
void raytracer_term(raytracer rt);
raytracer raytracer_test(config c);



#endif
