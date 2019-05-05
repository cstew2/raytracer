#include <stdio.h>
#include <stdlib.h>

#include "render/render_file.h"
#include "render/cuda_raytracer.cuh"
#include "render/cpu_raytracer.h"
#include "debug/debug.h"


void file_render(raytracer rt)
{	
	render(rt);
		
	char filename[] = "output.ppm";
	write_ppm_file(filename, rt.canvas);
}

void write_ppm_file(char *filename, canvas c)
{
	log_msg(INFO, "Writing single frame of raytracer output to %s\n", filename);
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P6\n%d\n%d\n255\n", c.width, c.height);

	colour col;
	uint8_t r;
	uint8_t g;
	uint8_t b;
	for(int i=0; i < c.height; i++) {
		for(int j=0; j < c.width; j++) {
			col = canvas_get_pixel(c, j, i);
			r = get_channel(col, RED);
			g = get_channel(col, GREEN);
			b = get_channel(col, BLUE);
			fwrite(&r, 1, 1, fp);
			fwrite(&g, 1, 1, fp);
			fwrite(&b, 1, 1, fp);
		}
	}

	fclose(fp);
}
