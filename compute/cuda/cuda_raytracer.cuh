#ifndef __CUDA_RAYTRACER_CU__
#define __CUDA_RAYTRACER_CU__

#include "render/camera.hh"
#include "world/sphere.hh"
#include "world/plane.hh"
#include "render/ray.hh"
#include "math/vector.hh"
#include "compute/common_raytracer.hh"

#ifdef __NVCC__
extern "C" {
#endif	
#include "world/material.h"
#include "render/raytracer.h"

typedef struct {
	scene *objects_d;
	sphere *spheres_d;
	sphere *planes_d;
	sphere *triangles_d;
	sphere *lights_d;

	camera *cam_d;
	
	canvas *can_d;
	vec4 *screen_d;
}cuda_rt;
#ifdef __NVCC__
}
#endif

__device__ int cuda_trace(const ray r, const scene *objects, hit_info *hi);
__device__ vec4 cuda_cast_ray(const ray r, const scene *objects, const camera *cam);
__global__ void cuda_render_kernel(const scene *objects, camera *cam, canvas *can);

#ifdef __NVCC__
extern "C" {
#endif
cuda_rt *cuda_init(raytracer rt);
void cuda_term(cuda_rt *crt);
int cuda_render(raytracer rt, cuda_rt *crt);
#ifdef __NVCC__
}
#endif

#endif
