#include <string.h>
#include <stdlib.h>

#include "canvas.h"

canvas canvas_init(int width, int height)
{
	canvas cv;
	cv.width = width;
	cv.height = height;
	cv.screen = calloc(width * height, sizeof(colour));
	return cv;
}

void canvas_term(canvas c)
{
	free(c.screen);
}

void canvas_set_pixel(canvas c, int x, int y, colour col)
{
	c.screen[(c.height * x) +  y] = col;
}

colour canvas_get_pixel(canvas c, int x, int y)
{
        return c.screen[(c.height * x) +  y];
}

void canvas_update(canvas src, canvas dest)
{
	if(src.width != dest.width || src.height != dest.height) {
		canvas_term(dest);
		dest = canvas_init(src.width, src.height);
	}
	memcpy(src.screen, dest.screen, src.width * src.height);
}
