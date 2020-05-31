#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
	
#include "render/canvas.hh"
#include "debug/debug.h"

canvas canvas_init(int width, int height)
{
	log_msg(INFO, "Initializing Canvas\n");
	canvas cv;
	cv.width = width;
	cv.height = height;
	cv.screen = (vec4 *) calloc(width * height, sizeof(vec4));
	return cv;
}

void canvas_term(canvas c)
{
	log_msg(INFO, "Terminating Canvas\n");
	free(c.screen);
}

__host__ __device__ void canvas_set_pixel(canvas *c, int x, int y, vec4 colour)
{
	c->screen[(c->width * y) + x] = colour;
}
	
vec4 canvas_get_pixel(canvas c, int x, int y)
{
        return c.screen[(c.width * y) + x];
}

void canvas_update(canvas src, canvas dest)
{
	if(src.width != dest.width || src.height != dest.height) {
		canvas_term(dest);
		dest = canvas_init(src.width, src.height);
	}
	memcpy(src.screen, dest.screen, src.width * src.height);
}

#ifdef __cplusplus
}
#endif
