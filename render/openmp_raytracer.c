#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

#include "render/openmp_raytracer.h"
#include "render/cpu_raytracer.h"

int openmp_render(const raytracer rt)
{	
	ray r;
	vec4 c;

        #pragma omp parallel for collapse(2) private(r, c)
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(&rt.camera, x, y);
		        c = cpu_cast_ray(r, rt);
			canvas_set_pixel(&rt.canvas, x, y, c);
			
		}
	}
	return 0;
}
