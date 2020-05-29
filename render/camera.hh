#ifndef __CAMERA_H__
#define __CAMERA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "math/vector.hh"
#include "render/ray.hh"
#include "main/cuda_check.h"

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

	vec3 w_p;
}camera;

camera camera_init(vec3 p, vec3 d, vec3 u, vec3 r, int width, int height, float fov);

void camera_right(camera *c, float speed);
void camera_left(camera *c, float speed);
void camera_forward(camera *c, float speed);
void camera_backward(camera *c, float speed);
void camera_up(camera *c, float speed);
void camera_down(camera *c, float speed);
void camera_rotate(camera *c, float pitch, float yaw);
__host__ __device__ ray generate_ray(camera c, int x, int y);

#ifdef __cplusplus
}
#endif
	
#endif
