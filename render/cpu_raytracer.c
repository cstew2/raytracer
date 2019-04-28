#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "math/vector.h"
#include "math/constants.h"
#include "math/math.h"
#include "render/ray.h"

#include "render/cpu_raytracer.h"

#include "world/scene.h"

colour BAD_COLOUR = 0xA043D3;

vec3 reflection(vec3 incident, vec3 normal)
{
	//r = i - 2(n . i)*n
	return vec3_sub(incident, vec3_scale(normal, (vec3_dot(normal, incident), 2)));
}

int trace(ray r, raytracer rt, hit_info *hi)
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
				hi->hit_n = vec3_new(0, 0, 0);
				hi->hit_p = vec3_new(0, 0, 0);
				hi->hit_m = rt.objects->planes[i].m;
				t_near = t;
			}
		}
	}
	
	return (t_near != 1.0/0.0);
}

colour cast_ray(ray r, raytracer rt)
{
	hit_info hi;
	hi.hit_c = BAD_COLOUR;
	if(trace(r, rt, &hi)) {
		for(int i=0; i < 7; i++){
			//printf("%d ", hit_c);
			if(hi.hit_m.reflection) {
				//vec3 r = reflect(r.direction, hi.hit_n);
				//hi.hit_c += 0.8; //fix this
			}
		}
	}
	return hi.hit_c;
}

int render(raytracer rt)
{
	ray r;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(rt.camera, y, x);
			canvas_set_pixel(rt.canvas, x, y, cast_ray(r, rt));
		}
	}
	return 0;
}
