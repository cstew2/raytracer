#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

config_element *parser(char *file_data)
{
	config_element *config_options = malloc(sizeof(config_element) * CONFIG_COUNT);
	memcpy(config_options, config_defaults, sizeof(config_element) * CONFIG_COUNT);
	
	for(int i=0; file_data[i] != '\0'; i++) {
		char *name = malloc(sizeof(char) * BUFFER_LEN);
		char *value = malloc(sizeof(char) * BUFFER_LEN);
		while(file_data[i] != '=') {
			if(file_data[i] != ' ') {
				name += file_data[i];
			}
			i++;
		}
		i++;
		while(file_data[i] != '\n') {
			if(file_data[i] != ' ') {
				name += file_data[i];
			}
			i++;
		}
		i++;
		for(int j=0; j < CONFIG_COUNT; j++) {
			if(strncmp(config_options[j].name, name, strlen(name))){
				if(config_options[j].t == INT) {
					config_options[j].i= atoi(value);
				} else if(config_options[j].t == FLOAT) {
					config_options[j].f = atof(value);
				} else if(config_options[j].t == STRING) {
					strncpy(config_options[j].s, value, BUFFER_LEN);
				}
				break;
			}
		}
	}
	return config_options;
}
