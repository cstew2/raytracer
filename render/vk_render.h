#ifndef __VK_RENDER__
#define __VK_RENDER__

#include <vulkan/vulkan.h> 
#include <GLFW/glfw3.h>

#include "render/raytracer.h"

VkInstance instance;
VkSurfaceKHR *surface;


void vk_init(void);
void vk_cleanup(void);

void vk_input(void);
void vk_render(void);

void create_instance(void);


#endif
