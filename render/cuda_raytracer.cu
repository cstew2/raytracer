#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "cuda_raytracer.cuh"

int cuda_trace(const ray r, const raytracer rt, hit_info *hi)
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

colour cuda_cast_ray(const ray r, const raytracer rt)
{
	hit_info hi;
	cuda_trace(r, rt, &hi);
	
	if(hi.hit_m.diffuse != 0) {
		
	}
	
	return hi.hit_c;
}

int cuda_render(const raytracer rt)
{
	ray r;
	colour c;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(rt.camera, x, y);
		        c = cuda_cast_ray(r, rt);
			canvas_set_pixel(rt.canvas, x, y, c);
		}
	}
	return 0;
}
