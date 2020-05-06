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

float ambient(const float a, const float k_a)
{
	
	return 0;
}

vec3 specular_reflection(const vec3 i, const vec3 n)
{
	//r=i-n(2(n.i))
	return vec3_sub(i, vec3_scale(n, 2*vec3_dot(i, n)));
}

vec3 refraction(const vec3 i, const vec3 n, const float a)
{
	//
	float cosi = clamp(-1, 1, vec3_dot(i, n));
	float etai = 1;
	float etat = a;
	vec3 N = n;
	
	if (cosi < 0) {
		cosi = -cosi;
	}
	else {
		float temp = etai;
		etai = etat;
		etat = temp;
		N = vec3_scale(n, -1);
	}
	float eta = etai / etat; 
	float k = 1 - eta * eta * (1 - cosi * cosi);

	if(k > 0) {
		return vec3_add(vec3_scale(i, eta), vec3_scale(N, (eta * cosi - sqrtf(k))));
	}
	return vec3_origin;
}

float fresnel(const vec3 i, const vec3 n, const float a, const float b)
{
	
	return 0;
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
	
	if(hi.hit_m.diffuse != 0) {
		
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
