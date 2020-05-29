#ifndef __SCENE_H__
#define __SCENE_H__

#include "world/plane.hh"
#include "world/sphere.hh"
#include "world/triangle.hh"
#include "world/light.h"

typedef struct {
	int sphere_count;
	int plane_count;
	int triangle_count;
	int light_count;
	
	sphere *spheres;
	plane *planes;
	triangle *triangles;
	light *lights;
}scene;

scene *scene_init(void);
void scene_term(scene *s);
void add_spheres(scene *s, int sphere_count, sphere *to_add);
void add_planes(scene *s, int plane_count, plane *to_add);
void add_triangles(scene *s, int triangle_count, triangle *to_add);
void add_lights(scene *s, int light_count, light *to_add);

#endif
