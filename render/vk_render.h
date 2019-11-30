#ifndef __VK_RENDER_H__
#define __VK_RENDER_H__

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "render/raytracer.h"

void vk_realtime_render(raytracer rt);

GLFWwindow *vk_glfw_init(config c);
VkInstance vk_init(void);
void vk_cleanup(GLFWwindow *window, VkInstance instance);

void vk_render(GLFWwindow *window);
void vk_update(GLFWwindow *window);
void vk_input(GLFWwindow *window);

VkInstance create_instance(void);

#endif
