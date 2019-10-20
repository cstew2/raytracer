#include <stdlib.h>

#include "render/raytracer.h"
#include "debug/debug.h"

raytracer raytracer_init(config c, camera cam, scene *objs)
{
	log_msg(INFO, "Initializing Raytracing Construct\n");
	raytracer r;
	r.config = c;
	r.camera = cam;
	r.objects = objs;
	r.canvas = canvas_init(c.width, c.height);
	
	return r;
}

void raytracer_term(raytracer rt)
{
	log_msg(INFO, "Terminating Raytracing Construct\n");
	scene_term(rt.objects);
	//segfault with nvidia driver?
	//canvas_term(rt.canvas);
}
		
raytracer raytracer_test(config c)
{

	camera cam = camera_init(vec3_new(20.0, 20.0, 20.0),
				 vec3_new(0.0, 0.0, 3.0),
				 vec3_new(0.0, 0.0, 1.0),
				 c.width,
				 c.height,
				 c.fov);
	
	int plane_count = 3;
	plane *planes = calloc(sizeof(plane), 3);
	planes[0] = new_plane(vec3_new(0.0, 0.0, 0.0),
			      vec3_new(0.0, 0.0, 1.0),
			      colour_new(255, 0, 0), matte);
	planes[1] = new_plane(vec3_new(0.0, 0.0, 0.0),
			      vec3_new(0.0, 1.0, 0.0),
			      colour_new(0, 255, 0), matte);
	planes[2] = new_plane(vec3_new(0.0, 0.0, 0.0),
			      vec3_new(1.0, 0.0, 0.0),
			      colour_new(0, 0, 255), matte);

	int sphere_count = 3;
	sphere *spheres = calloc(sizeof(sphere), 3);
	spheres[0] = new_sphere(vec3_new(5.0, 5.0, 5.0),
				2,
				colour_new(255, 0, 0), matte);
	spheres[1] = new_sphere(vec3_new(5.0, 10.0, 6.0),
				3,
				colour_new(0, 255, 0), glass);
	spheres[2] = new_sphere(vec3_new(10.0, 5.0, 7.0),
				4,
				colour_new(0, 0, 255), glass);

	int triangle_count = 0;
	triangle *triangles = NULL;
	/*
	triangle *triangles = calloc(sizeof(triangle), 2);
	triangles[0] = new_triangle(vec3_new(0.0, 0.0, 0.0),
				    vec3_new(1.0, 0.0, 0.0),
				    vec3_new(0.0, 1.0, 0.0),
				    colour_new(100, 100, 100));
	triangles[1] = new_triangle(vec3_new(0.0, 0.0, 0.0),
				    vec3_new(0.0, 1.0, 0.0),
				    vec3_new(1.0, 0.0, 0.0),
				    colour_new(100, 100, 100));
	*/
	int light_count = 0;
	light *lights = NULL;
	/*
	light *lights = calloc(sizeof(light), 2);
	lights[0] = light_new(colour_new(0.0, 0.0, 0.0),
			      100.0,
			      vec3_new(0.0, 0.0, 100.0),
			      vec3_new(0.0, 0.0, -1.0));
	lights[1] = light_new(colour_new(0.0, 0.0, 0.0),
			      100.0,
			      vec3_new(0.0, 0.0, -10.0),
			      vec3_new(0.0, 0.0, -1.0));
	*/
	
	scene *objs = scene_init();
	add_spheres(objs, sphere_count, spheres);
	add_planes(objs, plane_count, planes);
	add_triangles(objs, triangle_count, triangles);
	add_lights(objs, light_count, lights);

	free(planes);
	free(spheres);
	//free(triangles);
	//free(lights);
	
	raytracer r = raytracer_init(c, cam, objs);
		
	return r;
}