#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "render/cuda_raytracer.cuh"

__device__ int cuda_trace(const ray r, const scene *objects, bool shadow, hit_info *hi)
{
	float t_near = 1.0/0.0;
	float t;
	
	for(int i=0; i < objects->sphere_count; i++) {
		if(sphere_intersect(r, objects->spheres[i], &t)) {
			if(t_near > t) {
				t_near = t;
				sphere_hit(r, t, objects->spheres[i], hi);
			}
		}
	}

	for(int i=0; i < objects->plane_count; i++) {
		if(plane_intersect(r, objects->planes[i], &t)) {
			if(t_near > t) {
				t_near = t;
				plane_hit(r, t, objects->planes[i], hi);
			}
		}	
	}
	
	return t_near != 1.0/0.0;
}

__device__ vec4 cuda_cast_ray(const ray r, const scene *objects, const camera *cam)
{
	hit_info hi;
	cuda_trace(r, objects, false, &hi);

	vec4 c = vec4_new(0.0, 0.0, 0.0, 0.0);
	
	if(hi.hit_m.ambient > 0) {
		c = vec4_add(c, ambient(hi.hit_c, hi.hit_m.ambient));
	}

	if(hi.hit_m.diffuse  > 0 ||
	   hi.hit_m.specular > 0) {
		for(int l=0; l < objects->light_count; l++) {
			light l_i = objects->lights[l];
			vec3 l = vec3_scale(l_i.direction, -1.0);
			if(hi.hit_m.diffuse > 0) {
				vec4 d_colour = diffuse(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
							l, l_i.intensity, l_i.c);
				c = vec4_add(c, d_colour);
			}
		
			if(hi.hit_m.specular > 0) {
			
				vec4 s_colour = specular_reflection(hi.hit_n, hi.hit_c, hi.hit_m.diffuse,
								    l, l_i.intensity, l_i.c,
								    cam->position, 1.0);
				//c = vec4_add(c, s_colour);
			}
		}
	}
	
	return c;
}

__global__ void cuda_render_kernel(const scene *objects, camera *cam, canvas *can)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	ray r = generate_ray(cam, x, y);
	vec4 c = cuda_cast_ray(r, objects, cam);
	canvas_set_pixel(can, x, y, c);
}


#ifdef __NVCC__
extern "C" {
#endif

int cuda_render(raytracer rt)
{
	scene *objects_d = NULL;
	sphere *spheres_d = NULL;
	sphere *planes_d = NULL;
	sphere *triangles_d = NULL;
	sphere *lights_d = NULL;

	cudaMalloc(&objects_d, sizeof(scene));
	cudaMalloc(&spheres_d, sizeof(sphere)*rt.objects->sphere_count);
	cudaMalloc(&planes_d, sizeof(plane)*rt.objects->plane_count);
	cudaMalloc(&triangles_d, sizeof(triangle)*rt.objects->triangle_count);
	cudaMalloc(&lights_d, sizeof(light)*rt.objects->light_count);
	
	cudaMemcpy(objects_d, rt.objects,
		   sizeof(scene), cudaMemcpyHostToDevice);
	cudaMemcpy(spheres_d, rt.objects->spheres,
		   sizeof(sphere)*rt.objects->sphere_count, cudaMemcpyHostToDevice);
	cudaMemcpy(planes_d, rt.objects->planes,
		   sizeof(plane)*rt.objects->plane_count, cudaMemcpyHostToDevice);
	cudaMemcpy(triangles_d, rt.objects->triangles,
		   sizeof(triangle)*rt.objects->triangle_count, cudaMemcpyHostToDevice);
	cudaMemcpy(lights_d, rt.objects->lights,
		   sizeof(light)*rt.objects->light_count, cudaMemcpyHostToDevice);	
	
	cudaMemcpy(&(objects_d->spheres), &spheres_d, sizeof(sphere *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(objects_d->planes), &planes_d, sizeof(plane *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(objects_d->triangles), &triangles_d, sizeof(triangle *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(objects_d->lights), &lights_d, sizeof(light *), cudaMemcpyHostToDevice);
	
	camera *cam_d = NULL;
	cudaMalloc(&cam_d, sizeof(camera));
	cudaMemcpy(cam_d, &rt.camera, sizeof(camera), cudaMemcpyHostToDevice);

	canvas *can_d = NULL;
	vec4 *screen_d = NULL;

	cudaMalloc(&can_d, sizeof(canvas));
	cudaMalloc(&screen_d, sizeof(vec4)*rt.canvas.width*rt.canvas.height);
	
	cudaMemcpy(can_d, &rt.canvas, sizeof(canvas), cudaMemcpyHostToDevice);

	cudaMemcpy(screen_d, rt.canvas.screen,
		   sizeof(vec4)*rt.canvas.width*rt.canvas.height, cudaMemcpyHostToDevice);
	cudaMemcpy(&(can_d->screen), &screen_d, sizeof(vec4 *), cudaMemcpyHostToDevice);
	
	dim3 threads(8, 8);
	dim3 blocks(rt.canvas.width/threads.x,  
		    rt.canvas.height/threads.y);
	
	cuda_render_kernel<<<blocks, threads>>>(objects_d, cam_d, can_d);

	cudaMemcpy(rt.canvas.screen, screen_d,
		   sizeof(vec4)*rt.canvas.width*rt.canvas.height, cudaMemcpyDeviceToHost);

	write_ppm_file("frame.pnm", rt.canvas);
	
	cudaFree(screen_d);
	cudaFree(can_d);
	cudaFree(cam_d);
	cudaFree(lights_d);
	cudaFree(triangles_d);
	cudaFree(planes_d);
	cudaFree(spheres_d);
	cudaFree(objects_d);
	
	return 0;
}
	
#ifdef __NVCC__
}
#endif	
