#ifndef __RENDER_H__
#define __RENDER_H__

extern int WINDOW_HEIGHT;
extern int WINDOW_WIDTH;

void glfw_error_callback( int error, const char *description);
void glfw_window_size_callback(GLFWwindow *, int width, int height);


#endif
