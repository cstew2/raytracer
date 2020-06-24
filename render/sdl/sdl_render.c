#include <stdio.h>
#include <stdlib.h>

#include "math/constants.h"
#include "render/sdl/sdl_render.h"
#include "debug/debug.h"

#include "compute/cuda/cuda_raytracer.cuh"
#include "compute/pthread/pthread_raytracer.h"
#include "compute/openmp/openmp_raytracer.h"
#include "compute/cpu/cpu_raytracer.h"

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
	log_msg(INFO, "Initializing SDL\n");
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		log_msg(ERROR, "%s\n", SDL_GetError());
		return NULL;
	} 	
		
	sdl_data *data= malloc(sizeof(sdl_data));
	data->quit = false;
	data->rt = rt;

	log_msg(INFO, "Creating window\n");
        data->window = SDL_CreateWindow("SDL Raytracer",
				     SDL_WINDOWPOS_CENTERED,
				     SDL_WINDOWPOS_CENTERED,
				     rt.config.width,
				     rt.config.height,
				     SDL_WINDOW_SHOWN |
				     SDL_WINDOW_MOUSE_CAPTURE);
	if(!data->window) {
		log_msg(ERROR, "%s\n", SDL_GetError());
		return NULL;
	}

	log_msg(INFO, "Creating renderer\n");
	data->renderer = SDL_CreateRenderer(data->window,
					 -1,
					 SDL_RENDERER_ACCELERATED |
					 SDL_RENDERER_TARGETTEXTURE);
	if(!data->renderer) {
		log_msg(ERROR, "%s\n", SDL_GetError());
		return NULL;
	}
	
	log_msg(INFO, "Creating texture to render to\n");
	data->texture = SDL_CreateTexture(data->renderer,
				       SDL_PIXELFORMAT_RGBA8888,
				       SDL_TEXTUREACCESS_TARGET,
				       rt.config.width,
				       rt.config.height);
	if(!data->texture) {
		log_msg(ERROR, "%s", SDL_GetError());
		return NULL;
	}

	SDL_SetRelativeMouseMode(true);
        SDL_GetRelativeMouseState((int *)&rt.state.last_x, (int *)&rt.state.last_y);       
	SDL_SetWindowGrab(data->window, true);

	data->pixel_buffer = calloc(sizeof(uint32_t), rt.config.width * rt.config.height);
	
	data->frame_count = 0;
	data->previous_time = 0;
	
	return data;
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
	//get next from from raytracing renderer
	//cpu_render(data->r, NULL);
	threaded_render(data->rt, NULL);
	//cuda_render(data->r, NULL);
	//openmp_render(data->rt, NULL);

	//need to convert from 4 floats to packed 32 bit int
	for(int j=0; j < data->rt.canvas.height; j++) {
		for(int i=0; i < data->rt.canvas.width; i++) {
			vec4 c = canvas_get_pixel(data->rt.canvas, i, j);
			data->pixel_buffer[(j * data->rt.canvas.width) + i] =
				(unsigned char)(c.x*255) << 24 | //R
				(unsigned char)(c.y*255) << 16 | //G
				(unsigned char)(c.z*255) << 8  | //B
				(unsigned char)(c.w*255);        //A
		}
	}
	
	int pitch = data->rt.canvas.width * 4;
	SDL_UpdateTexture(data->texture, NULL, (void *)data->pixel_buffer, pitch);
		
	SDL_RenderClear(data->renderer);
	SDL_SetRenderDrawColor(data->renderer, 255, 255, 255, 255);

        SDL_RenderCopyEx(data->renderer, data->texture, NULL, NULL, 0, NULL,
			 SDL_FLIP_VERTICAL);
        SDL_RenderPresent(data->renderer);
}

void sdl_update(sdl_data *data)
{
	sdl_update_fps_counter(data);
	
	if(data->rt.state.forward) {
		camera_forward(&data->rt.camera, data->rt.state.speed);
	}
	if(data->rt.state.left) {
		camera_left(&data->rt.camera, data->rt.state.speed);
	}
	if(data->rt.state.right) {
		camera_right(&data->rt.camera, data->rt.state.speed);
	}
	if(data->rt.state.back) {
		camera_backward(&data->rt.camera, data->rt.state.speed);
	}
	if(data->rt.state.up) {
		camera_up(&data->rt.camera, data->rt.state.speed);	
	}
	if(data->rt.state.down) {
		camera_down(&data->rt.camera, data->rt.state.speed);	
	}
	log_msg(DEBUG, "position: %f, %f, %f\n direction: %f, %f, %f\n up: %f, %f, %f\n right: %f, %f, %f\n",
		data->rt.camera.position.x,
		data->rt.camera.position.y,
		data->rt.camera.position.z,
		data->rt.camera.direction.x,
		data->rt.camera.direction.y,
		data->rt.camera.direction.z,
		data->rt.camera.up.x,
		data->rt.camera.up.y,
		data->rt.camera.up.z,
		data->rt.camera.right.x,
		data->rt.camera.right.y,
		data->rt.camera.right.z);
	log_msg(DEBUG, "f:%d, b:%d, l:%d. r:%d, u:%d, d:%d\n",
		data->rt.state.forward, data->rt.state.back, data->rt.state.left,
		data->rt.state.right, data->rt.state.up, data->rt.state.down);
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
				data->rt.state.forward = true;
				break;
			case SDLK_a:
				data->rt.state.left = true;
				break;
			case SDLK_s:
				data->rt.state.back = true;
				break;
			case SDLK_d:
				data->rt.state.right = true;
				break;
			case SDLK_q:
				data->rt.state.up = true;
				break;
			case SDLK_e:
				data->rt.state.down = true;
				break;
			default:
				break;
			}
			break; 

		case SDL_KEYUP:
			switch(event.key.keysym.sym) {
			case SDLK_w:
				data->rt.state.forward = false;
				break;
			case SDLK_a:
				data->rt.state.left = false;
				break;
			case SDLK_s:
				data->rt.state.back = false;
				break;
			case SDLK_d:
				data->rt.state.right = false;
				break;
			case SDLK_q:
				data->rt.state.up = false;
				break;
			case SDLK_e:
				data->rt.state.down = false;
				break;
			default:
				break;
			}
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
			}
			else if(event.wheel.y < 0) {
				log_msg(INFO, "Mouse Wheel down\n");
			}

			if(event.wheel.x > 0) {
				log_msg(INFO, "Mouse Wheel left\n");
			}
			else if(event.wheel.x < 0) {
				log_msg(INFO, "Mouse Wheel right\n");
			}
			break;
			
			/* Mouse movement handling */
		case SDL_MOUSEMOTION:
		        ;//needed for delcarations after case label
			int xoffset = event.motion.xrel;
			int yoffset = -event.motion.yrel;
			
			xoffset *= data->rt.state.sensitivity;
			yoffset *= data->rt.state.sensitivity;
	
			data->rt.state.yaw   += xoffset;
			data->rt.state.pitch += yoffset;
	
			if (data->rt.state.pitch > 89.9f) {
				data->rt.state.pitch = 89.9f;
			}
			if (data->rt.state.pitch < -89.9f) {
				data->rt.state.pitch = -89.9f;
			}
			
			log_msg(DEBUG, "mouse pitch: %f, mouse yaw: %f\n",
				data->rt.state.pitch, data->rt.state.yaw);
			camera_rotate(&data->rt.camera, data->rt.state.pitch, data->rt.state.yaw);
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
}

void sdl_update_fps_counter(sdl_data *data)
{
	char tmp[64];

	double current_time = SDL_GetTicks();
	double elapsed_time = current_time - data->previous_time;
	data->frame_count++;
	if(elapsed_time > 0.1f) {
		double fps = ((double)data->frame_count/elapsed_time)*1000;
		sprintf(tmp, "SDL - Raytracer @ fps: %.2f", fps);
		SDL_SetWindowTitle(data->window, tmp);
		
		data->previous_time = current_time;
		data->frame_count = 0;
	}
}
