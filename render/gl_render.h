#ifndef __GL_RENDER_H__
#define __GL_RENDER_H__

#include <GLFW/glfw3.h>
#include <stdbool.h>

#include "render/render.h"


int gl_render(void);
int gl_init(void);
void gl_cleanup(void);
void gl_input(void);

void update_fps_counter(GLFWwindow *w);
int load_texture(canvas *c, GLuint *tex);
void log_gl_params(void);
bool gl_log_err(const char *message, ...);
bool gl_log(const char *message, ...);
bool restart_gl_log(void);
GLchar const *load_shader(const char *filename);

#endif
