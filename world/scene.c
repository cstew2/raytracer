#include <stdlib.h>
#include <string.h>

#include "world/scene.h"

scene *scene_init(void)
{
	scene *s = malloc(sizeof(scene));
	s->sphere_count = 0;
	s->plane_count = 0;
	s->triangle_count = 0;
	s->light_count = 0;

	s->spheres = NULL;
	s->planes = NULL;
	s->triangles = NULL;
	s->lights = NULL;
	
	return s;
}

void add_spheres(scene *s, int sphere_count, sphere *to_add)
{
	s->spheres = realloc(s->spheres, s->sphere_count + sphere_count);
	memcpy(s->spheres + (sizeof(sphere) * s->sphere_count), to_add, sphere_count);
	s->sphere_count += sphere_count;
}

void add_planes(scene *s, int plane_count, plane *to_add)
{
	s->planes = realloc(s->planes, s->plane_count + plane_count);
	memcpy(s->planes + (sizeof(plane) * s->plane_count), to_add, plane_count);
	s->plane_count += plane_count;
}

void add_triangles(scene *s, int triangle_count, triangle *to_add)
{
	s->triangles = realloc(s->triangles, s->triangle_count + triangle_count);
	memcpy(s->triangles + (sizeof(triangle) * s->triangle_count), to_add, triangle_count);
	s->triangle_count += triangle_count;
}

void add_lights(scene *s, int light_count, light *to_add)
{
	s->lights = realloc(s->lights, s->light_count + light_count);
	memcpy(s->lights + (sizeof(light) * s->light_count), to_add, light_count);
	s->light_count += light_count;
}

void scene_term(scene *s)
{
	free(s->spheres);
	free(s->planes);
	free(s->triangles);
	free(s->lights);
}
