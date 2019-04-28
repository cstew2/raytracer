#include <stdlib.h>
#include <stdio.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "vk_render.h"

#include "debug/debug.h"

int WINDOW_WIDTH;
int WINDOW_HEIGHT;
GLFWwindow *window;


void vk_realtime_render(raytracer rt)
{
	
}

void vk_init(config *c)
{
	log_msg(INFO, "Starting GLFW: %s\n", glfwGetVersionString());
	if (!glfwInit()) {
		log_msg(ERROR, "Could not start GLFW3\n");
		return;
	}
	
	if(c->fullscreen) {
		log_msg(INFO, "Using fullscreen mode\n");
		GLFWmonitor* mon = glfwGetPrimaryMonitor();
		const GLFWvidmode* vmode = glfwGetVideoMode(mon);
		window = glfwCreateWindow (vmode->width, vmode->height,
					   "Vulkan - Voxel Raytracer", mon, NULL);
	}
	else {
		log_msg(INFO, "Using windowed mode\n");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(c->width, c->height,
					  "Vulkan - Voxel Raytracer", NULL, NULL);
		
	}
	WINDOW_WIDTH = c->width;
	WINDOW_HEIGHT = c->height;
	if (!window) {
		log_msg(ERROR, "Could not open window with GLFW3\n");
		glfwTerminate();
		return;
	}

	
}

void vk_term(void)
{

}

void vk_render(void)
{

}

void vk_update(void)
{

}

void vk_input(void)
{

}

VkInstance createInstance() {
	VkInstance instance;
	
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Raytracer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "cudaray";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
	        log_msg(ERROR, "Cannot create vulkan instance");
        }
	return instance;
}
