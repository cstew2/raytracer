#include <stdlib.h>

#include "render/raytracer.h"
#include "debug/debug.h"

static state default_state = {
	.forward = false,
	.left = false,
	.right = false,
	.back = false,
	.up = false,
	.down = false,

	.last_x = 0,
	.last_y = 0, 
	.yaw = 0.0f,
	.pitch = 0.0f,
	.speed = 1.0f,
	.sensitivity = 0.5f
};

raytracer raytracer_init(config c, camera cam, scene *objs)
{
	log_msg(INFO, "Initializing Raytracing Construct\n");
	raytracer r;
	r.config = c;
	r.camera = cam;
	r.objects = objs;
	r.canvas = canvas_init(c.width, c.height);
	r.state = default_state;
	return r;
}

void raytracer_term(raytracer rt)
{
	log_msg(INFO, "Terminating Raytracing Construct\n");
	scene_term(rt.objects);
	
	canvas_term(rt.canvas);
}
		
raytracer raytracer_test(config c)
{
	log_msg(INFO, "Initializing Raytracing Test scene\n");
	camera cam = camera_init(vec3_new(-4, 7.0, 7.0),
				 vec3_new(0.5, 0.5, 0.0),
				 vec3_new(0.0, 0.0, 1.0),
				 vec3_new(0.0, 1.0, 0.0),
				 c.width,
				 c.height,
				 c.fov);
	
	int plane_count = 2;
	plane *planes = calloc(sizeof(plane), plane_count);
	planes[0] = plane_new(vec3_new(0.0, 0.0, 100.0),
			      vec3_new(0.0, 0.0, 1.0),
			      vec4_new(0.5, 0.5, 0.9, 1.0),
			      flat);
	planes[1] = plane_new(vec3_new(0.0, 0.0, 0.0),
			      vec3_new(0.0, 0.0, -1.0),
			      vec4_new(0.5, 0.9, 0.5, 1.0),
			      flat);
			      
	int sphere_count = 7;
	sphere *spheres = calloc(sizeof(sphere), sphere_count);
	spheres[0] = sphere_new(vec3_new(5.0, 5.0, 5.0),
				2,
				vec4_new(1.0, 0.0, 0.0, 1.0),
				shiny);
	spheres[1] = sphere_new(vec3_new(5.0, 10.0, 6.0),
				3,
				vec4_new(0.0, 1.0, 0.0, 1.0),
				glass);
	spheres[2] = sphere_new(vec3_new(10.0, 5.0, 7.0),
				4,
				vec4_new(0.0, 0.0, 1.0, 1.0),
				glass);
	spheres[3] = sphere_new(vec3_new(0.0, 0.0, 7.0),
				4,
				vec4_new(1.0, 1.0, 1.0, 1.0),
				glass);
	
	spheres[4] = sphere_new(vec3_new(5.0, 0.0, 3.0),
				4,
				vec4_new(1.0, 0.0, 0.0, 1.0),
				glass);
	spheres[5] = sphere_new(vec3_new(10.0, -7.0, 3.0),
				3,
				vec4_new(0.0, 1.0, 0.0, 1.0),
				matte);
	spheres[6] = sphere_new(vec3_new(20.0, 7.0, 20.0),
				1,
				vec4_new(1.0, 1.0, 1.0, 1.0),
				flat);

	
	int triangle_count = 2;
	triangle *triangles = calloc(sizeof(triangle), triangle_count);
	triangles[0] = triangle_new(vec3_new(0.0, 0.0, 0.0),
				    vec3_new(1.0, 0.0, 0.0),
				    vec3_new(0.0, 1.0, 0.0),
				    vec4_new(1.0, 1.0, 1.0, 1.0));
	triangles[1] = triangle_new(vec3_new(0.0, 0.0, 0.0),
				    vec3_new(0.0, 1.0, 0.0),
				    vec3_new(1.0, 0.0, 0.0),
				    vec4_new(1.0, 1.0, 1.0, 1.0));
	
	int light_count = 1;
	light *lights = calloc(sizeof(light), light_count);
	lights[0] = light_new(vec4_new(1.0, 1.0, 1.0, 1.0),
			      1.0,
			      1.0,
			      vec3_new(20.0, 7.0, 20.0),
			      vec3_new(0.0, 0.0, -1.0));
	
	
	scene *objs = scene_init();
	add_spheres(objs, sphere_count, spheres);
	add_planes(objs, plane_count, planes);
	add_triangles(objs, triangle_count, triangles);
	add_lights(objs, light_count, lights);

	free(planes);
	free(spheres);
	free(triangles);
	free(lights);
	
	raytracer r = raytracer_init(c, cam, objs);
		
	return r;
}
