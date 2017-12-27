#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "main/main.h"
#include "debug/debug.h"
#include "main/config.h"
#include "render/render.h"
#include "render/gl_render.h"
#include "render/vk_render.h"
#include "render/raytracer.h"

int main(int argc, char **argv)
{
	log_init();
	char *filename = NULL;
	char *config = NULL;
	bool opengl =  true;
	bool vulkan = false;
	
	if(argc > 1) {
		for(int i=0; i < argc; i++) {
			if(!strncmp(argv[i], "-f", 2)) {
				i++;
				filename = malloc(sizeof(char) * strlen(argv[i] + 1));
				strncpy(filename, argv[i], strlen(argv[i] + 1));
			}
			else if(!strncmp(argv[i], "-c", 2)) {
				i++;
				config = malloc(sizeof(char) * strlen(argv[i] + 1));
				strncpy(filename, argv[i], strlen(argv[i] + 1));
			}
			else if(!strncmp(argv[i], "-vk", 3)) {
				vulkan = true;
				opengl = false;
			}
			else if(!strncmp(argv[i], "-gl", 3)) {
				opengl = true;
			}
		}
	}
	if(config != NULL) {
		parser(get_file(config));
	}

	if(opengl && !vulkan) {
		gl_main_loop();
	}
	else if(!opengl && vulkan) {
		vk_main_loop();
	}
	else {
		log_msg(ERROR, "You cannot select both OpenGL(-gl) and Vulkan(-vk) as renderers\n");
	}
	
	return 0;
}

void gl_main_loop(void)
{
	gl_init();
	while(!glfwWindowShouldClose(window)) {
		gl_input();
		update();
		gl_render();
	}
}

void vk_main_loop(void)
{
	vk_init();
	while(!glfwWindowShouldClose(window)) {
		vk_input();
		update();
		vk_render();
	}
}
