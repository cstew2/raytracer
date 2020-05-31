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

void scene_term(scene *s)
{
	free(s->spheres);
	free(s->planes);
	free(s->triangles);
	free(s->lights);
	free(s);
}

void add_spheres(scene *s, int sphere_count, sphere *to_add)
{
	s->spheres = calloc(sizeof(sphere), sphere_count);
	s->spheres = memcpy(s->spheres, to_add, sphere_count * sizeof(sphere));
	s->sphere_count += sphere_count;
}

void add_planes(scene *s, int plane_count, plane *to_add)
{
	s->planes = calloc(sizeof(plane), plane_count);
	s->planes = memcpy(s->planes, to_add, plane_count * sizeof(plane));
	s->plane_count += plane_count;
}

void add_triangles(scene *s, int triangle_count, triangle *to_add)
{
	s->triangles = calloc(sizeof(triangle), triangle_count);
	s->triangles = memcpy(s->triangles, to_add, triangle_count * sizeof(triangle));
	s->triangle_count += triangle_count;
}

void add_lights(scene *s, int light_count, light *to_add)
{
	s->lights = calloc(sizeof(light), light_count);
	s->lights = memcpy(s->lights, to_add, light_count * sizeof(light));
	s->light_count += light_count;
}


