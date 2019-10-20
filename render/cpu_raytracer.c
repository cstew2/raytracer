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

int cpu_trace()
{
	
}

colour cpu_cast_ray()
{
	
}

int cpu_render(raytracer rt)
{
	ray r;
	for(int y=0; y < rt.canvas.height; y++) {
		for(int x=0; x < rt.canvas.width; x++) {
			r = generate_ray(rt.camera, y, x);
			colour c = cast_ray(r, rt);
			canvas_set_pixel(rt.canvas, x, y, c);
		}
	}
	return 0;
}
