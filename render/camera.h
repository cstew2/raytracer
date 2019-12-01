#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "math/vector.h"
#include "render/ray.h"

typedef struct {
	vec3 position;
	vec3 direction;
	vec3 up;
	vec3 right;
		
	float fov;
	float aspect_ratio;
	float aperture;
	float near_focus;
	float far_focus;
	float tanfov;
	
	int width;
	int height;
}camera;

camera camera_init(vec3 p, vec3 d, vec3 u, vec3 r, int width, int height, float fov);

void camera_right(camera *c, float speed);
void camera_left(camera *c, float speed);
void camera_forward(camera *c, float speed);
void camera_backward(camera *c, float speed);
void camera_up(camera *c, float speed);
void camera_down(camera *c, float speed);
void camera_rotate(camera *c, float pitch, float yaw);

ray generate_ray(camera c, int x, int y);

#endif
