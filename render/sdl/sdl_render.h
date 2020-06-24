#ifndef __SDL_RENDER_H__
#define __SDL_RENDER_H__

#include <SDL2/SDL.h>

#include "render/raytracer.h"

typedef struct {
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	raytracer rt;

	uint32_t *pixel_buffer;
	
	int window_width;
	int window_height;

	bool quit;

	float previous_time;
	int frame_count;

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

void sdl_update_fps_counter(sdl_data *data);

#endif
