#ifndef __FB_RENDER_H__
#define __FB_RENDER_H__

#include "render/raytracer.h"
#include "main/config.h"

static const char *DEV_FB = "/dev/fb0";
static const char *DEV_TTY = "/dev/tty";

void fb_realtime_render(raytracer rt);
int fb_init(config c);
void fb_render(void);
void fb_input(void);
void fb_update(void);
void fb_cleanup(void);

#endif
