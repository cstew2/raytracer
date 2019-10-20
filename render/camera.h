#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "math/vector.h"
#include "render/ray.h"

typedef struct {
	vec3 position;
	vec3 direction;
	vec3 up;
		
	float fov;
	float aspect_ratio;
	float aperture;
	float near_focus;
	float far_focus;
	float tanfov;
	
	int width;
	int height;
}camera;

camera camera_init(vec3 p, vec3 d, vec3 u, int width, int height, float fov);

camera camera_rotate_z(camera c, float angle);
camera camera_rotate_y(camera c, float angle);
camera camera_rotate_x(camera c, float angle);

ray generate_ray(camera c, int x, int y);

#endif
