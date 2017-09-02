#ifndef _PAINT_H_
#define _PAINT_H_

#include <GLFW/glfw3.h>
#include <stdbool.h>

#include "world/colour.h"

typedef struct {
	int width;
	int height;
	colour *screen;
}canvas;

GLFWwindow *window;

canvas new_canvas(int width, int height, colour *c);

int gl_render(void);
int gl_init(void);
void gl_cleanup(void);
void gl_input(void);

	
int load_texture(canvas *c, GLuint *tex);
void update_fps_counter(GLFWwindow *w);
void log_gl_params(void);
void glfw_window_size_callback(GLFWwindow *, int width, int height);
void glfw_error_callback(int error, const char *description);
bool gl_log_err(const char *message, ...);
bool gl_log(const char *message, ...);
bool restart_gl_log(void);

#endif
