#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "render/cuda_raytracer.cuh"

__device__ int cuda_trace(const ray r, const raytracer *rt, bool shadow, hit_info *hi)
{
	float t_near = 1.0/0.0;
	float t;
	
	for(int i=0; i < rt->objects->sphere_count; i++) {
		if(sphere_intersect(r, rt->objects->spheres[i], &t)) {
			if(t_near > t) {
				t_near = t;
				sphere_hit(r, t, rt->objects->spheres[i], hi);
			}
		}
	}

	for(int i=0; i < rt->objects->plane_count; i++) {
		if(plane_intersect(r, rt->objects->planes[i], &t)) {
			if(t_near > t) {
				t_near = t;
				plane_hit(r, t, rt->objects->planes[i], hi);
			}
		}	
	}
	
	return t_near != 1.0/0.0;
}

__device__ vec4 cuda_cast_ray(const ray r, raytracer *rt)
{
	hit_info hi;
	cuda_trace(r, rt, false, &hi);

	vec4 c = vec4_new(0.0, 0.0, 0.0, 0.0);
	
	if(hi.hit_m.ambient > 0) {
		c = vec4_add(c, ambient(hi.hit_c, hi.hit_m.ambient));
	}

	if(hi.hit_m.diffuse  > 0 ||
	   hi.hit_m.specular > 0) {
		for(int l=0; l < rt->objects->light_count; l++) {
			light l_i = rt->objects->lights[l];
			vec3 l = vec3_scale(l_i.direction, -1.0);
			if(hi.hit_m.diffuse > 0) {
				vec4 d_colour = diffuse(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
							l, l_i.intensity, l_i.c);
				c = vec4_add(c, d_colour);
			}
		
			if(hi.hit_m.specular > 0) {
			
				vec4 s_colour = specular_reflection(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
								    l, l_i.intensity, l_i.c,
								    rt->camera.position, 1.0);
				//c = vec4_add(c, s_colour);
			}
		}
	}
	
	return c;
}

__global__ void cuda_render_kernel(raytracer *rt)
{
	uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	ray r;
	vec4 c;
	
	r = generate_ray(rt->camera, x, y);
	c = cuda_cast_ray(r, rt);
	canvas_set_pixel(rt->canvas, x, y, c);
}

int cuda_render(raytracer rt)
{
	raytracer *rt_d;
	cudaMalloc(&rt_d, sizeof(raytracer));
	cudaMemcpy(rt_d, &rt, sizeof(raytracer), cudaMemcpyHostToDevice);

	dim3 threads(8, 8);
	dim3 blocks(rt.canvas.width/threads.x,  
		    rt.canvas.height/threads.y);
	
	cuda_render_kernel<<<blocks, threads>>>(rt_d);

	
	
	free(rt_d);
	
	return 0;
}
