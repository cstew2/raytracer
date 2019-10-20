#ifndef __VK_RENDER_H__
#define __VK_RENDER_H__

#include "render/raytracer.h"

void vk_realtime_render(raytracer rt);

GLFWwindow *vk_init(config c);
void vk_cleanup(GLFWwindow *window);

void vk_render(GLFWwindow *window);
void vk_update(GLFWwindow *window);
void vk_input(GLFWwindow *window);

#endif
