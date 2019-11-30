#include <stdlib.h>
#include <stdio.h>

#include "vk_render.h"

#include "debug/debug.h"

int WINDOW_WIDTH;
int WINDOW_HEIGHT;

raytracer r;

void vk_realtime_render(raytracer rt)
{
	r = rt;
	GLFWwindow *window = vk_glfw_init(rt.config);
	VkInstance instance = vk_init();
	
	log_msg(INFO, "Starting main game loop\n");
	while(!glfwWindowShouldClose(window)) {
		vk_input(window);
		vk_update(window);
		vk_render(window);
	}
	vk_cleanup(window, instance);
}

GLFWwindow *vk_glfw_init(config c)
{
	log_msg(INFO, "Initializing Vulkan rendering setup\n");
	// start GL context and O/S window using the GLFW helper library
	log_msg(INFO, "Starting GLFW: %s\n", glfwGetVersionString());
	// register the error call-back function that we wrote, above
	if (!glfwInit()) {
		log_msg(ERROR, "Could not start GLFW3\n");
		return NULL;
	}
		
	GLFWwindow *window = NULL;
	if(c.fullscreen) {
		log_msg(INFO, "Using fullscreen mode\n");
		GLFWmonitor* mon = glfwGetPrimaryMonitor ();
		const GLFWvidmode* vmode = glfwGetVideoMode (mon);
		window = glfwCreateWindow (vmode->width, vmode->height,
					   "Vulkan - Voxel Raytracer", mon, NULL);
	}
	else {
		log_msg(INFO, "Using windowed mode\n");
		window = glfwCreateWindow(c.width, c.height,
					  "Vulkan - Voxel Raytracer", NULL, NULL);
	}
	
	if (!window) {
		log_msg(ERROR, "Could not open window with GLFW3\n");
		glfwTerminate();
		return NULL;
	}
	glfwMakeContextCurrent(window);

		
	return window;
}

VkInstance vk_init(void)
{
	VkInstance instance = create_instance();
	
	
	return instance;
}

void vk_cleanup(GLFWwindow *window, VkInstance instance)
{
	log_msg(INFO, "Terminating Vulkan Rendering setup\n");

	vkDestroyInstance(instance, NULL);
	
	glfwDestroyWindow(window);
       	glfwTerminate();
}

void vk_render(GLFWwindow *window)
{

}

void vk_update(GLFWwindow *window)
{

}

void vk_input(GLFWwindow *window)
{

}



VkInstance create_instance(void)
{
	VkInstance instance;
	
        VkApplicationInfo appInfo;
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Raytracer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "cudaray";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo;
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
