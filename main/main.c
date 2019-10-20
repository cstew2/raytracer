#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "main/main.h"
#include "debug/debug.h"

#include "render/raytracer.h"
#include "render/gl_render.h"
#include "render/vk_render.h"
#include "render/sdl_render.h"

#include "render/render_file.h"
#include "main/config.h"


int main(int argc, char **argv)
{
	log_init();
	char *filename = NULL;
	char *config_path = NULL;
	config c;

	bool threads = false;
	bool cuda = false;

	bool picture = false;
	bool opengl = false;
	bool vulkan = false;
	bool sdl = false;

	//check args
	if(argc > 1) {
		for(int i=0; i < argc; i++) {
			if(!strncmp(argv[i], "-f", 2)) {
				i++;
				if(argv[i]) {
					filename = malloc(sizeof(char) * strlen(argv[i] + 1));
					strcpy(filename, argv[i]);
				}
				else {
					log_msg(WARN, "You passed the -f flag without any parameter" \
						"after it, it will be ignored\n");
				}
			}
			else if(!strncmp(argv[i], "-c", 2)) {
				i++;
				log_msg(INFO, "Loading configuration file: %s\n", argv[i]);
				config_path = malloc(sizeof(char) * strlen(argv[i] + 1));
				strcpy(config_path, argv[i]);
			}
			else if(!strncmp(argv[i], "-p", 2)) {
				picture = true;
			}
			else if(!strncmp(argv[i], "-gl", 3)) {
				opengl = true;
			}
			else if(!strncmp(argv[i], "-vk", 3)) {
				vulkan = true;
			}
			else if(!strncmp(argv[i], "-sdl", 4)) {
				sdl = true;
			}
			else if(!strncmp(argv[i], "-cuda", 5)) {
				cuda = true;
			}
		}
	}

	//decide what to do with args 
	if(config_path != NULL) {
		c = parser(get_file(config_path));
	}
	else {
		c = default_config();
	}

	raytracer rt = raytracer_test(c);
	
	if(picture) {
		file_render(rt, filename);
	}
	else {
		if(opengl && vulkan && sdl) {
			log_msg(WARN, "You passed too many arguments for the render\n");
		}
		else if(opengl && !vulkan && !sdl) {
			gl_realtime_render(rt);
		}
		else if(!opengl && vulkan && !sdl) {
			vk_realtime_render(rt);
		}
		else if(!opengl && !vulkan && sdl) {
			sdl_realtime_render(rt);
		}
		else if(!(opengl && vulkan && sdl)) {
			log_msg(WARN, "No argument selected for the renderer, defaulting to opengl\n");
			gl_realtime_render(rt);
		}
		else {
			log_msg(WARN, "More than two arguments supplied for the renderer, defaulting to opengl\n");
			gl_realtime_render(rt);
		}
	}
	
	raytracer_term(rt);
	log_term();
	if(filename != NULL) {
		free(filename);
	}
	if(config_path != NULL) {
		free(config_path);
	}
	
	return 0;
}

void print_help(void)
{
	printf("cudaray usage: cudaray [OPTIONS]...\n\n"
	       "Options\n"
	       "-d \t debug on\n"
	       "-c \t pass a config file\n"
	       "-f \t open \"FILE\"\n"
	       "-gl \t use opengl to render"
	       "-vk \t use vulkan to render"
	       "-sdl \t use SDL to render"
	       "-p \t render single image to pnm file");
}

