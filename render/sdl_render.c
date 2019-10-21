#include <stdio.h>
#include <stdlib.h>

#include "math/constants.h"

#include "render/sdl_render.h"

#include "debug/debug.h"

void sdl_realtime_render(raytracer rt)
{
	sdl_data *r = sdl_init(rt);
	while(!r->quit) {
		sdl_input(r);
		sdl_update(r);
		sdl_render(r);
	}
	sdl_term(r);
}

sdl_data *sdl_init(raytracer rt)
{
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		return NULL;
	} 	

	sdl_data *r = malloc(sizeof(sdl_data));
	r->quit = false;
	r->rt = rt;
	
        r->window = SDL_CreateWindow("SDL Raytracer",
				     SDL_WINDOWPOS_CENTERED,
				     SDL_WINDOWPOS_CENTERED,
				     rt.config.width,
				     rt.config.height,
				     0);
	
	r->renderer = SDL_CreateRenderer(r->window,
					 -1,
					 SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
	
	r->texture = SDL_CreateTexture(r->renderer,
				       SDL_PIXELFORMAT_BGRA8888,
				       SDL_TEXTUREACCESS_STREAMING,
				       rt.config.width,
				       rt.config.height);

	return r;
}

void sdl_term(sdl_data *data)
{
	SDL_DestroyRenderer(data->renderer);
	SDL_DestroyWindow(data->window);
	free(data);
	SDL_Quit();
}

void sdl_resize(sdl_data *data)
{
	SDL_GetWindowSize(data->window, &data->window_width, &data->window_height);
	if(data->window_width == 0 || data->window_height == 0) {
		log_msg(ERROR, "SDL Window dimensions 0\n");
		exit(-1);
	}
	canvas_term(data->rt.canvas);
	data->rt.canvas = canvas_init(data->window_width, data->window_height);
}

void sdl_render(sdl_data *data)
{
	int pitch = data->rt.canvas.width * 4;
	SDL_LockTexture(data->texture, NULL, (void **)&data->rt.canvas.screen, &pitch);
	
        SDL_RenderCopy(data->renderer, data->texture, NULL, NULL);
        SDL_RenderPresent(data->renderer);
	
}

void sdl_update(sdl_data *data)
{

}

void sdl_input(sdl_data *data)
{
	SDL_Event event;
	while(SDL_PollEvent(&event) != 0) {
		switch(event.type) {
			/* Hitting window buttons */
		case SDL_QUIT:
			log_msg(INFO, "Received SDL_QUIT event, now closing\n");
			data->quit = true;
			break;
			/* keyboard handling */
		case SDL_KEYDOWN:
			switch(event.key.keysym.sym) {
			case SDLK_ESCAPE:
				log_msg(INFO, "Escape Key hit, now closing\n");
				data->quit = true;
				break;
			case SDLK_w:
				log_msg(INFO, "Key W down\n");
				data->rt.camera.position.x -= 1;
				break;
			case SDLK_a:
				log_msg(INFO, "Key A down\n");
				data->rt.camera.position.y += 1;
				break;
			case SDLK_s:
				log_msg(INFO, "Key S down\n");
				data->rt.camera.position.x += 1;
				break;
			case SDLK_d:
				log_msg(INFO, "Key D down\n");
				data->rt.camera.position.y -= 1;
				break;
			default:
				break;
			}
			printf("position x:%f, y:%f, z:%f\n",
			       data->rt.camera.position.x,
			       data->rt.camera.position.y,
			       data->rt.camera.position.z);
			break; 
			
			/* Mouse button Handling */
		case SDL_MOUSEBUTTONDOWN:
			switch (event.button.button)
			{
			case SDL_BUTTON_LEFT:
				log_msg(INFO, "Mouse Left Click\n");
				break;
			case SDL_BUTTON_RIGHT:
				log_msg(INFO, "Mouse Right Click\n");
				break;
			case SDL_BUTTON_MIDDLE:
				log_msg(INFO, "Mouse Middle Click\n");
				break;
			default:
				break;
			}
			break;

			/* Mouse wheel handling */ 
		case SDL_MOUSEWHEEL:
			if(event.wheel.y > 0) {
				log_msg(INFO, "Mouse Wheel up\n");
				data->rt.camera.position.z -= 1;
			}
			else if(event.wheel.y < 0) {
				log_msg(INFO, "Mouse Wheel down\n");
				data->rt.camera.position.z += 1;
			}

			if(event.wheel.x > 0) {
				log_msg(INFO, "Mouse Wheel left\n");
			}
			else if(event.wheel.x < 0) {
				log_msg(INFO, "Mouse Wheel right\n");
			}
			printf("position x:%f, y:%f, z:%f\n",
			       data->rt.camera.position.x,
			       data->rt.camera.position.y,
			       data->rt.camera.position.z);
			break;
			/* Mouse movement handling */
		case SDL_MOUSEMOTION:
			//log_msg(INFO, "Mouse at (%d, %d)\n", mousex, mousey);
			
			//rotation camera about x or y
			if(event.motion.x > data->mousex) {
				//data->rt.camera = camera_rotate_z(data->rt.camera, PI/(360.0));
			}
			//rotation camera about x or y
			else if(event.motion.x < data->mousex) {
				//data->rt.camera = camera_rotate_z(data->rt.camera, -PI/(360.0));
			}
			//rotate camera right about z
			else if(event.motion.y > data->mousey) {
			}
			//rotate camera left about z
			else if(event.motion.y < data->mousey) {
			}
			printf("lookat x:%f, y:%f, z:%f\n",
			       data->rt.camera.direction.x,
			       data->rt.camera.direction.y,
			       data->rt.camera.direction.z);
			data->mousex = event.motion.x;
			data->mousey = event.motion.y;
			break;
			/* window events*/
		case SDL_WINDOWEVENT:
			if(event.window.event == SDL_WINDOWEVENT_RESIZED) {
				log_msg(INFO, "Window resized\n");
				sdl_resize(data);
			}
			break;

		default:
			break;
		}
	}
	//log_msg(INFO, "Camera x:%f y:%f z:%f\n", cam->position.x, cam->position.y, cam->position.z);
}
