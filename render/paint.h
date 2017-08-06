#ifndef _PAINT_H_
#define _PAINT_H_

#include <GLFW/glfw3.h>

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

void glfw_error_callback(int error, const char* description);
int gl_log_err(const char* message, ...);
int restart_gl_log(void);
int gl_log(const char* message, ...);
void glfw_window_size_callback(GLFWwindow* w, int width, int height);

#endif
