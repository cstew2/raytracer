#ifndef __CANVAS_H__
#define __CANVAS_H__

#include "math/vector.hh"
#include "main/cuda_check.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int width;
	int height;
	vec4 *screen;
}canvas;

canvas canvas_init(int width, int height);
void canvas_term(canvas);
__host__ __device__ void canvas_set_pixel(canvas *c, int x, int y, vec4 colour);
vec4 canvas_get_pixel(canvas c, int x, int y);
void canvas_update(canvas src, canvas dest);

#ifdef __cplusplus
}
#endif
	
#endif
