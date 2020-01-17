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

vec3 reflection(const vec3 i, const vec3 n)
{
	//r=i-2(n.i)n
	return vec3_sub(i, vec3_scale(n, 2*vec3_dot(i, n)));
}

int refraction(const vec3 i, const vec3 n, const float a, vec3 *r)
{
	//n.sin(theta)=n'.sin(theta')
	vec3 ni = vec3_normalize(i);
	float dt = vec3_dot(ni, n);
	float disc = 1.0 - a*a*(1*dt*dt);
	if(disc > 0) {
		*r = vec3_sub(vec3_scale(vec3_sub(ni, vec3_scale(n, dt)), a), vec3_scale(n, disc));
		return true;
	}
	return false;
}

int fresnel(const vec3 i, const vec3 n, const float a, vec3 *r)
{
	
}

int cpu_trace(const ray r, const raytracer rt, hit_info *hi)
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

colour cpu_cast_ray(const ray r, const raytracer rt)
{
	hit_info hi;
	cpu_trace(r, rt, &hi);
	if() {
		
	}
	else if() {
		
	}
	else if() {
		
	}
	
	return hi.hit_c;
}

int cpu_render(const raytracer rt)
{
	ray r;
	colour c;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(rt.camera, x, y);
		        c = cpu_cast_ray(r, rt);
			canvas_set_pixel(rt.canvas, x, y, c);
		}
	}
	return 0;
}
