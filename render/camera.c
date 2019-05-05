#include <math.h>

#include "render/camera.h"

camera camera_init(vec3 p, vec3 d, vec3 u, float fov, float aspect, float aperture, float focus)
{
	camera c;
	c.position = p;
	c.direction = d;
	c.up = u;
	c.right = vec3_cross(d, u);
	
	c.fov = fov;
	c.aspect = aspect;
	c.aperture = aperture;
	c.focus = focus;

	c.half_width = 2 * atan(tan((fov/2) * aspect));
	c.half_height = c.half_width * aspect;
	c.bottom_left = vec3_new(c.half_width, c.half_height, 1.0);
	
	return c;
}

camera camera_rotate_z(camera c, float angle)
{
        float x = (cosf(angle) * (c.direction.x-c.position.x) - sinf(angle) * (c.direction.x-c.position.x)) + c.position.x;
	float y = (sinf(angle) * (c.direction.y-c.position.y) + cosf(angle) * (c.direction.y-c.position.y)) + c.position.y;
	float z = c.direction.z;

	c.direction = vec3_normalise(vec3_new(x, y, z));
	return c;
}


camera camera_rotate_y(camera c, float angle)
{
        float x = (cosf(angle) * (c.direction.x-c.position.x) - sinf(angle) * (c.direction.x-c.position.x)) + c.position.x;
	float y = c.direction.y;
	float z = (sinf(angle) * (c.direction.z-c.position.z) + cosf(angle) * (c.direction.z-c.position.z)) + c.position.z;

	c.direction = vec3_normalise(vec3_new(x, y, z));
	return c;
}

camera camera_rotate_x(camera c, float angle)
{
        float x = c.direction.x;
	float y = (cosf(angle) * (c.direction.y-c.position.y) - sinf(angle) * (c.direction.y-c.position.y)) + c.position.x;
	float z = (sinf(angle) * (c.direction.z-c.position.z) + cosf(angle) * (c.direction.z-c.position.z)) + c.position.z;

	c.direction = vec3_normalise(vec3_new(x, y, z));
	return c;
}


ray generate_ray(camera c, int i, int j)
{
	return ray_init(c.position, vec3_new(0,0,0));
}
