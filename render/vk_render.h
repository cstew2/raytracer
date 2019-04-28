#ifndef __VK_RENDER_H__
#define __VK_RENDER_H__

#include "render/raytracer.h"

void vk_realtime_render(raytracer rt);

void vk_init(config *c);
void vk_term(void);

void vk_render(void);
void vk_update(void);
void vk_input(void);

#endif
