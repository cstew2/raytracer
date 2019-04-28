#ifndef __RENDER_H__
#define __RENDER_H__

#include "world/colour.h"
#include <GLFW/glfw3.h>

int WINDOW_HEIGHT;
int WINDOW_WIDTH;

GLFWwindow *window;

void glfw_error_callback(int error, const char *description);


#endif
