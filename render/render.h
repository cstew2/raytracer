#ifndef __RENDER_H__
#define __RENDER_H__

#include "world/colour.h"
#include <GLFW/glfw3.h>

extern int WINDOW_HEIGHT;
extern int WINDOW_WIDTH;

typedef struct {
	int width;
	int height;
	colour *screen;
}canvas;

GLFWwindow *window;

canvas new_canvas(int width, int height, colour *c);

void glfw_error_callback( int error, const char *description);
void glfw_window_size_callback(GLFWwindow *, int width, int height);


#endif
