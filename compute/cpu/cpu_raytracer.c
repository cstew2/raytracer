#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "math/vector.hh"
#include "math/constants.h"
#include "math/math.h"
#include "render/ray.hh"
#include "world/scene.h"

#include "compute/common_raytracer.hh"
#include "compute/cpu/cpu_raytracer.h"

int cpu_trace(const ray r, const raytracer rt, bool shadow, hit_info *hi)
{
	float t_near = 1.0/0.0;
	float t;
	
	for(int i=0; i < rt.objects->sphere_count; i++) {
		if(sphere_intersect(r, rt.objects->spheres[i], &t)) {
			if(t_near > t) {
				t_near = t;
				sphere_hit(r, t, rt.objects->spheres[i], hi);
			}
		}
	}

	for(int i=0; i < rt.objects->plane_count; i++) {
		if(plane_intersect(r, rt.objects->planes[i], &t)) {
			if(t_near > t) {
				t_near = t;
				plane_hit(r, t, rt.objects->planes[i], hi);
			}
		}	
	}
	
	return t_near != 1.0/0.0;
}

vec4 cpu_cast_ray(const ray r, const raytracer rt)
{
	hit_info hi;
	cpu_trace(r, rt, false, &hi);

	vec4 c = vec4_new(0.0, 0.0, 0.0, 0.0);
	
	if(hi.hit_m.ambient > 0) {
		c = vec4_add(c, ambient(hi.hit_c, hi.hit_m.ambient));
	}

	if(hi.hit_m.diffuse  > 0 ||
	   hi.hit_m.specular > 0) {
		for(int l=0; l < rt.objects->light_count; l++) {
			light l_i = rt.objects->lights[l];
			vec3 l = vec3_scale(l_i.direction, -1.0);
			if(hi.hit_m.diffuse > 0) {
				vec4 d_colour = diffuse(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
							l, l_i.intensity, l_i.c);
				c = vec4_add(c, d_colour);
			}
		
			if(hi.hit_m.specular > 0) {
			
				vec4 s_colour = specular_reflection(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
								    l, l_i.intensity, l_i.c,
								    rt.camera.position, 1.0);
				//c = vec4_add(c, s_colour);
			}
		}
	}
	
	return c;
}

int cpu_render(const raytracer rt, void *cuda_rt)
{
	ray r;
	vec4 c;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(&rt.camera, x, y);
		        c = cpu_cast_ray(r, rt);
			canvas_set_pixel(&rt.canvas, x, y, c);
		}
	}
	return 0;
}
