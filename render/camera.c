#include <math.h>

#include "render/camera.h"

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
	c.tanfov = (deg2rad(c.fov/2));
	
	c.width = width;
	c.height = height;
	
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
	/*
	vec3 forward;
	forward.x = cosf(deg2rad(yaw)) * cosf(deg2rad(pitch));
	forward.y = sin(deg2rad(yaw)) * cosf(deg2rad(pitch));
	forward.z = sin(deg2rad(pitch));
	c->direction = vec3_normalize(forward);
	c->up = vec3_new(0.0, 0.0, 1.0);
	c->right = vec3_normalize(vec3_cross(c->direction, c->up));
	*/
}

ray generate_ray(camera c, int x, int y)
{
	vec3 ray_direction;
	ray_direction.x = 0;
	ray_direction.y = (2 * ((x + 0.5) / c.width) - 1) * c.tanfov * c.aspect_ratio;
	ray_direction.z = -(1 - 2 * ((y + 0.5) / c.height)) * c.tanfov;
	
	ray_direction = vec3_normalize(vec3_add(c.direction, ray_direction));
	return ray_init(c.position, ray_direction);
}
