#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "math/vector.h"
#include "render/ray.h"

typedef struct {
	vec3 position;
	vec3 direction;
	vec3 up;
	vec3 right;

	float half_height;
	float half_width;
	vec3 bottom_left;
	
	float fov;
	float aspect;
	float aperture;
	float focus;
}camera;

camera camera_init(vec3 p, vec3 d, vec3 u, float fov, float aspect, float aperture, float focus);

camera camera_rotate_z(camera c, float angle);
camera camera_rotate_y(camera c, float angle);
camera camera_rotate_x(camera c, float angle);

ray generate_ray(camera c, int i, int j);

#endif
