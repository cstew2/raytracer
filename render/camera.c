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
	c.direction = d;
	c.up = u;

	c.fov = fov;
	c.aspect_ratio = width/height;
	c.tanfov = (deg2rad(c.fov/2));
	
	c.width = width;
	c.height = height;
	
	return c;
}

camera camera_rotate_z(camera c, float angle)
{
	
	return c;
}


camera camera_rotate_y(camera c, float angle)
{
	
	return c;
}

camera camera_rotate_x(camera c, float angle)
{
	
	return c;
}


ray generate_ray(camera c, int x, int y)
{
	vec3 ray_direction;
	ray_direction.x = (2 * ((x + 0.5) / c.width) - 1) * c.tanfov * c.aspect_ratio;
	ray_direction.y = (1 - 2 * ((y + 0.5) / c.height)) * c.tanfov;
	ray_direction.z = -1;
	return ray_init(c.position, vec3_normalise(ray_direction));
}
