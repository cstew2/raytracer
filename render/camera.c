#include <math.h>

#include "render/camera.h"

#include "math/constants.h"
#include "math/math.h"
#include "debug/debug.h"

camera camera_init(vec3 p, vec3 d, vec3 u, int width, int height, float fov)
{
	log_msg(INFO, "Initializing Camera\n");
	camera c;
	c.position = p;
	c.direction = vec3_normalise(d);
	c.up = vec3_normalise(u);
	c.right = vec3_normalise(vec3_cross(c.direction, c.up));

	c.fov = fov;
	c.aspect_ratio = width/height;
	c.tanfov = (deg2rad(c.fov/2));
	
	c.width = width;
	c.height = height;
	
	return c;
}

void camera_right(camera *c, float speed)
{
	float cam_y = c->position.y;
	c->position = vec3_add(c->position, vec3_scale(vec3_normalise(vec3_cross(c->direction, c->up)), speed));
	c->position.y = cam_y;
}

void camera_left(camera *c, float speed)
{
	float cam_y = c->position.y;
	c->position = vec3_sub(c->position, vec3_scale(vec3_normalise(vec3_cross(c->direction, c->up)), speed));
	c->position.y = cam_y;
}

void camera_forward(camera *c, float speed)
{
	float cam_y = c->position.y;
	c->position = vec3_add(c->position, vec3_scale(c->direction, speed));
	c->position.y = cam_y;
}

void camera_backward(camera *c, float speed)
{
	float cam_y = c->position.y;
	c->position = vec3_sub(c->position, vec3_scale(c->direction, speed));
	c->position.y = cam_y;
}

void camera_up(camera *c, float speed)
{
	c->position = vec3_sub(c->position, vec3_new(0, speed, 0));	
}

void camera_down(camera *c, float speed)
{
	c->position = vec3_add(c->position, vec3_new(0, speed, 0));
}

void camera_rotate(camera *c, float pitch, float yaw)
{
	vec3 front;
	front.x = cosf(deg2rad(yaw)) * cosf(deg2rad(pitch));
	front.y = sinf(deg2rad(pitch));
	front.z = sinf(deg2rad(yaw)) * cosf(deg2rad(pitch));
	c->direction = vec3_normalise(front);
	c->up = vec3_new(0.0, 1.0, 0.0);
}

ray generate_ray(camera c, int x, int y)
{
	vec3 ray_direction;
	ray_direction.x = (2 * ((x + 0.5) / c.width) - 1) * c.tanfov * c.aspect_ratio;
	ray_direction.y = (1 - 2 * ((y + 0.5) / c.height)) * c.tanfov;
	ray_direction.z = 0;
	
	ray_direction = vec3_normalise(vec3_add(c.direction, ray_direction));
	return ray_init(c.position, ray_direction);
}
