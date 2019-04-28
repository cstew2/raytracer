#include <GLFW/glfw3.h>

#include "render/render.h"

#include "debug/debug.h"

void glfw_error_callback(int error, const char *description)
{
	log_msg(ERROR, "GLFW ERROR: code %i msg: %s\n", error, description);
}

