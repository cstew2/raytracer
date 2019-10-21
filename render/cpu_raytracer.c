#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "render/cpu_raytracer.h"

#include "math/vector.h"
#include "math/constants.h"
#include "math/math.h"
#include "render/ray.h"
#include "world/scene.h"

int cpu_trace(ray r, raytracer rt, hit_info *hi)
{
	float t_near = 1.0/0.0;
	float t;
	for(int i=0; i < rt.objects->sphere_count; i++) {
		if(sphere_intersect(r, rt.objects->spheres[i], &t)) {
			if(t_near > t) {
				hi->hit_c = rt.objects->spheres[i].c;
				hi->hit_n = vec3_new(0, 0, 0);
				hi->hit_p = vec3_new(0, 0, 0);
				hi->hit_m = rt.objects->spheres[i].m;
				t_near = t;
			}
		}
	}

	for(int i=0; i < rt.objects->plane_count; i++) {
		if(plane_intersect(r, rt.objects->planes[i], &t)) {
			if(t_near > t) {
				hi->hit_c = rt.objects->planes[i].c;
				hi->hit_n = rt.objects->planes[i].normal;
				hi->hit_p = vec3_new(0, 0, 0);
				hi->hit_m = rt.objects->planes[i].m;
				t_near = t;
			}
		}	
	}
	
	return t_near != 1.0/0.0;
}

colour cpu_cast_ray(ray r, raytracer rt)
{
	hit_info hi;
	hi.hit_c = colour_new(255, 255, 255);
	cpu_trace(r, rt, &hi);
	return hi.hit_c;
}

int cpu_render(raytracer rt)
{
	ray r;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(rt.camera, y, x);
			colour c = cpu_cast_ray(r, rt);
			canvas_set_pixel(rt.canvas, x, y, c);
		}
	}
	return 0;
}
