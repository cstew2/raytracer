#ifndef __SDL_RENDER_H__
#define __SDL_RENDER_H__

#include <SDL2/SDL.h>

#include "render/raytracer.h"
#include "render/cuda_raytracer.cuh"
#include "render/cpu_raytracer.h"

typedef struct {
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	raytracer rt;

	int window_width;
	int window_height;

	bool quit;

	int frame;

	int mousex;
	int mousey;

	unsigned int ticks;
	unsigned int last_ticks;
}sdl_data;

void sdl_realtime_render(raytracer rt);

sdl_data *sdl_init(raytracer rt);
void sdl_term(sdl_data *data);
void sdl_resize(sdl_data *data);

void sdl_render(sdl_data *data);
void sdl_update(sdl_data *data);
void sdl_input(sdl_data *data);

#endif