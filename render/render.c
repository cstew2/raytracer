#include <GLFW/glfw3.h>

#include "render/render.h"

#include "debug/debug.h"

int WINDOW_HEIGHT = 600;
int WINDOW_WIDTH = 800;

canvas new_canvas(int width, int height, colour *c)
{
	canvas cv;
	cv.width = width;
	cv.height = height;
	cv.screen = c;
	return cv;
}

void glfw_error_callback(int error, const char *description)
{
	log_msg(ERROR, "GLFW ERROR: code %i msg: %s\n", error, description);
}

void glfw_window_size_callback(GLFWwindow *w, int width, int height)
{
        glfwSetWindowSize(w, width, height);
        WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
	log_msg(INFO, "width: %i height: %i\n", width, height);
	/* update any perspective matrices used here */
}
