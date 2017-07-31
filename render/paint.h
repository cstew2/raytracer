#ifndef _PAINT_H_
#define _PAINT_H_

#include "world/colour.h"

typedef struct {
	int width;
	int height;
	colour *screen;
}canvas;

canvas new_canvas(int width, int height, colour *c);

void gl_init(void);
void init(void);
void render(void);

#endif
