#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "main/config.h"
#include "debug/debug.h"

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
	
	char *file_buffer = calloc(size, sizeof(char));
	fread(file_buffer, 1, size, f);
	fclose(f);

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
		name[0] = '\0';
		value[0] = '\0';
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
			while(isalnum(file_data[i]) && i < size) {
				value[value_count++] = file_data[i++];
			}

			if(strncmp(name, "fov", name_count)) {
				c.fov = atof(value);
			}
			else if(strncmp(name, "draw_distance", name_count)) {
				c.draw_distance = atof(value);
			}
			else if(strncmp(name, "width", name_count)) {
				c.width = atoi(value);
			}
			else if(strncmp(name, "height", name_count)) {
				c.height = atoi(value);
			}
			else if(strncmp(name, "fullscreen", name_count)) {
				c.fullscreen = strncmp(value, "true", value_count) ? true : false;
			}
			else if(strncmp(name, "fps", name_count)) {
				c.fullscreen = atoi(value);
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
