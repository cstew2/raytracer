#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "main/config.h"

static const config config_defaults = {
	//log level
	.log_level = ERROR,
	
	//platform rendering
	.raytracer_method = CPU,
	.render_method = OPENGL,
	
	//rendering
	.fov = 90.0,
	.draw_distance = 200.0,
	.max_depth = 5,
	.bias = 0.1,
	
	//window
	.width = 640,
	.height = 360,
	.fullscreen = false,
	.fps = 120
};

char *get_file(const char *filename)
{
	if(filename == NULL) {
		log_msg(INFO, "no file given to load\n");
		return NULL;
	}
	FILE *f = NULL;
	size_t size;
	f = fopen(filename, "r");
	if(f == NULL) {
		log_msg(INFO, "Improper path given to load file\n");
		return NULL;
	}
	
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	
	char *file_buffer = calloc(size+1, sizeof(char));
	size_t read_size = fread(file_buffer, 1, size, f);
	if(size != read_size){
		log_msg(WARN, "Could not read all of config file %s\n", filename);
	}
	fclose(f);

	file_buffer[size] = '\0';
	
	return file_buffer;
}

config parser(char *file_data)
{
	config c = config_defaults;

	size_t size = strlen(file_data);
	char *name = malloc(BUFFER_LEN);
	char *value = malloc(BUFFER_LEN);
	int name_count = 0;
	int value_count = 0;
	for(size_t i = 0; i < size; i++)
	{
		name_count = 0;
		value_count = 0;
		//comment line
		if(file_data[i] == '#') {
			while(file_data[i] != '\n' && i < size) {
				i++;
			}
		}

		//data line
		else if(isalnum(file_data[i])) {
			while(file_data[i] != ' ' && i < size) {
				name[name_count++] = file_data[i++];
			}
			while(!isalnum(file_data[i]) && i < size) {
				i++;
			}
			while(file_data[i] != '\n'  && i < size) {
				value[value_count++] = file_data[i++];
			}
			name[name_count] = '\0';
			value[value_count] = '\0';

			if(!strncmp(name, "log_level", name_count)) {
				if(!strncmp(value, "error", value_count)) { 
					c.log_level = ERROR;
				}
				else if(!strncmp(value, "warn", value_count)) { 
					c.log_level = WARN;
				}
				else if(!strncmp(value, "info", value_count)) { 
					c.log_level = INFO;
				}
				else if(!strncmp(value, "debug", value_count)) { 
					c.log_level = DEBUG;
				}
			}
			else if(!strncmp(name, "raytrace_method", name_count)) {
				if(!strncmp(value, "cpu", value_count)) { 
					c.raytracer_method = CPU;
				}
				else if(!strncmp(value, "openmp", value_count)) { 
					c.raytracer_method = OPENGL;
				}
				else if(!strncmp(value, "cuda", value_count)) { 
					c.raytracer_method = VULKAN;
				}
				else if(!strncmp(value, "opencl", value_count)) { 
					c.raytracer_method = SDL;
				}
			}
			else if(!strncmp(name, "render_method", name_count)) {
				if(!strncmp(value, "ppm", value_count)) { 
					c.render_method = PPM;
				}
				else if(!strncmp(value, "opengl", value_count)) { 
					c.render_method = OPENGL;
				}
				else if(!strncmp(value, "vulkan", value_count)) { 
					c.render_method = VULKAN;
				}
				else if(!strncmp(value, "sdl", value_count)) { 
					c.render_method = SDL;
				}
				else if(!strncmp(value, "linux_fb", value_count)) { 
					c.render_method = LINUX_FB;
				}
			}
			else if(!strncmp(name, "fov", name_count)) {
				c.fov = atof(value);
			}
			else if(!strncmp(name, "draw_distance", name_count)) {
				c.draw_distance = atof(value);
			}
			else if(!strncmp(name, "width", name_count)) {
				c.width = atoi(value);
			}
			else if(!strncmp(name, "height", name_count)) {
				c.height = atoi(value);
			}
			else if(!strncmp(name, "fullscreen", name_count)) {
				c.fullscreen = strncmp(value, "true", value_count) ? false : true;
			}
			else if(!strncmp(name, "fps", name_count)) {
				c.fps = atoi(value);
			}
		}
		//all other lines skipped
		
	}
	free(name);
	free(value);
	return c;
}

config default_config(void)
{
	return config_defaults;
}
