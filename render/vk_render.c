#include "vk_render.h"

#include "render/render.h"
#include "debug/debug.h"

void vk_init(void)
{
	log_msg(INFO, "starting GLFW: %s\n", glfwGetVersionString());
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		log_msg(ERROR, "could not start GLFW3\n");
		return;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);


	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vukan - Voxel Raytracer", NULL, NULL);
	if (!window) {
		log_msg(ERROR, "could not open window with GLFW3\n");
		glfwTerminate();
		return;
	}
	glfwSetWindowSizeCallback(window, glfw_window_size_callback);

	create_instance();
	
        glfwCreateWindowSurface(instance, window, NULL, surface);
	
}

void vk_input(void)
{
	glfwPollEvents();
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose(window, 1);
	}	
}

void vk_render(void)
{
	
}

void vk_cleanup(void)
{
	vkDestroyInstance(instance, NULL);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void create_instance(void) {
        VkApplicationInfo appInfo;
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
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
		log_msg(ERROR, "failed to create instance!");
        }
}
