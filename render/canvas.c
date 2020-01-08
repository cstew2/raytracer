#include <string.h>
#include <stdlib.h>

#include "render/canvas.h"
#include "debug/debug.h"

canvas canvas_init(int width, int height)
{
	log_msg(INFO, "Initializing Canvas\n");
	canvas cv;
	cv.width = width;
	cv.height = height;
	cv.screen = calloc(width * height, sizeof(colour));
	return cv;
}

void canvas_term(canvas c)
{
	log_msg(INFO, "Terminating Canvas\n");
	free(c.screen);
}

void canvas_set_pixel(canvas c, int x, int y, colour col)
{
	c.screen[(c.width * y) + x] = col;
}

colour canvas_get_pixel(canvas c, int x, int y)
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
