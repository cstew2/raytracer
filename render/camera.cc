#include <math.h>

#ifdef __NVCC__
extern "C" {
#endif	

#include "render/camera.hh"

#include "math/constants.h"
#include "math/math.h"
#include "debug/debug.h"

camera camera_init(vec3 p, vec3 d, vec3 u, vec3 r, int width, int height, float fov)
{
	log_msg(INFO, "Initializing Camera\n");
	camera c;
	c.position = p;
	c.direction = vec3_normalize(d);
	c.up = vec3_normalize(u);
	c.right = vec3_normalize(r);
	
	c.fov = fov;
	c.aspect_ratio = width/height;
	c.tanfov = tan(deg2rad(c.fov/2));
	
	c.width = width;
	c.height = height;

	c.w_p = vec3_add(vec3_sub(vec3_scale(c.right, -c.width/2),
				  vec3_scale(c.up, c.height/2)),
			 vec3_scale(c.direction, (c.height/2)/c.tanfov));
	
	return c;
}

void camera_right(camera *c, float speed)
{
	c->position = vec3_add(c->position, vec3_scale(c->right, speed));
}

void camera_left(camera *c, float speed)
{
	c->position = vec3_sub(c->position, vec3_scale(c->right, speed));
}

void camera_forward(camera *c, float speed)
{
	c->position = vec3_add(c->position, vec3_scale(c->direction, speed));
}

void camera_backward(camera *c, float speed)
{
	c->position = vec3_sub(c->position, vec3_scale(c->direction, speed));
}

void camera_up(camera *c, float speed)
{
	c->position = vec3_add(c->position, vec3_new(0, 0, speed));	
}

void camera_down(camera *c, float speed)
{
	c->position = vec3_sub(c->position, vec3_new(0, 0, speed));
}

void camera_rotate(camera *c, float pitch, float yaw)
{
	vec3 direction;
	
	direction.x = cos(deg2rad(yaw)) * cos(deg2rad(pitch));
	direction.y = cos(deg2rad(pitch)) * sin(deg2rad(yaw));
	direction.z = sin(deg2rad(pitch));
	
        c->direction = vec3_normalize(direction);
	c->right = vec3_normalize(vec3_cross(vec3_new(0.0, 0.0, 1.0), c->direction));
	c->up = vec3_normalize(vec3_cross(c->direction, c->right));

	c->w_p = vec3_add(vec3_sub(vec3_scale(c->right, -c->width/2),
				   vec3_scale(c->up, c->height/2)),
			  vec3_scale(c->direction, (c->height/2)/c->tanfov));
}



__host__ __device__ ray generate_ray(camera c, int x, int y)
{      
	vec3 ray_direction = vec3_normalize(vec3_add(vec3_add(vec3_scale(c.right, x),
							      vec3_scale(c.up, y)),
						     c.w_p));
	return ray_init(c.position, ray_direction);
}

#ifdef __NVCC__
}
#endif	
