#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "main/main.h"
#include "debug/debug.h"
#include "main/config.h"
#include "render/render.h"

int main(int argc, char **argv)
{
	log_init();
	char *filename = NULL;
	char *config = NULL;
	if(argc > 1) {
		for(int i=0; i < argc; i++) {
			if(strncmp(argv[i], "-f", 2)) {
				i++;
				filename = malloc(sizeof(char) * strlen(argv[i] + 1));
				strncpy(filename, argv[i], strlen(argv[i] + 1));
			}
			if(strncmp(argv[i], "-c", 2)) {
				i++;
				config = malloc(sizeof(char) * strlen(argv[i] + 1));
				strncpy(filename, argv[i], strlen(argv[i] + 1));
			}
		}
	}
	if(config != NULL) {
		parser(get_file(config));
	}
	
	main_loop();
	
	return 0;
}

void main_loop(void)
{
	gl_init();
	while(!glfwWindowShouldClose(window)) {
		input();
		update();
		render();
	}
}
void render(void)
{
	gl_render();
}

void update(void)
{
	gl_input();
}

void input(void)
{
	gl_input();
}
