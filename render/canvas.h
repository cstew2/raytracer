#ifndef __CANVAS_H__
#define __CANVAS_H__

#include "world/colour.h"

typedef struct {
	int width;
	int height;
	colour *screen;
}canvas;

canvas canvas_init(int width, int height);
void canvas_term(canvas);
void canvas_set_pixel(canvas c, int x, int y, colour col);
colour canvas_get_pixel(canvas c, int x, int y);
void canvas_update(canvas src, canvas dest);

#endif
