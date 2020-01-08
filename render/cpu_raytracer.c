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
	return vec3_sub(i, vec3_scale(n, 2*vec3_dot(i, n)));
}

vec3 refraction(const vec3 i, const vec3 n, const float a)
{
	float cosi = clamp(vec3_dot(i, n), -1, 1); 
	float etai = 1;
	float etat = ior; 
	if(cosi < 0) {
		cosi = -cosi;
		
	}
	else {
		etai = ior;
	        etat = 1;
	
	}
	float eta = etai / etat; 
	float k = 1 - eta * eta * (1 - cosi * cosi); 
	return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

float fresnel(const vec3 i, const vec3 n, const float a)
{
	float cosi = clamp(vec3_dot(i, n), -1, 1); 
	float etai = 1;
	float etat = ior; 
	if (cosi > 0) {
		std::swap(etai, etat);
	} 
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(maxf(0.0, 1 - cosi * cosi)); 
	// Total internal reflection
	if (sint >= 1) { 
		return = 1; 
	} 
	else { 
		float cost = sqrtf(maxf(0.f, 1 - sint * sint)); 
		cosi = fabsf(cosi); 
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost)); 
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost)); 
		return (Rs * Rs + Rp * Rp) / 2; 
	} 
	// As a consequence of the conservation of energy, transmittance is given by:
	// kt = 1 - kr;	
}

int cpu_trace(const ray r, const raytracer rt, hit_info *hi)
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

colour cpu_cast_ray(const ray r, const raytracer rt)
{
	hit_info hi;
	cpu_trace(r, rt, &hi);

	if(hi.hit_m.opaque) {
		
	}
	else {
		
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
