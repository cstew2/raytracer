#include <stdio.h>
#include <stdlib.h>

#include "render/cuda_raytracer.cuh"

#include "render/cpu_raytracer.h"
#include "debug/debug.h"
#include "render/render_file.h"

void file_render(raytracer rt, char *filename)
{	
	cpu_render(rt);
	write_ppm_file(filename, rt.canvas);
}

void write_ppm_file(char *filename, canvas c)
{
	log_msg(INFO, "Writing single frame of raytracer output to %s\n", filename);
	FILE *fp = fopen(filename, "wb");
	if(!fp) {
		log_msg(ERROR, "Could not write to file: %s\n", fp);
		return;
	}
	fprintf(fp, "P6\n%d\n%d\n255\n", c.width, c.height);

	vec4 col;
	unsigned char r;
	unsigned char g;
	unsigned char b;
	for(int y=c.height-1; y >= 0; y--) {
		for(int x=c.width-1; x >= 0; x--) {
			col = canvas_get_pixel(c, x, y);
			r = col.x * 255.999;
			g = col.y * 255.999;
			b = col.z * 255.999;
			fwrite(&r, 1, 1, fp);
			fwrite(&g, 1, 1, fp);
			fwrite(&b, 1, 1, fp);
		}
	}

	fclose(fp);
}
