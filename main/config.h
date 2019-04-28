#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdbool.h>

static const int BUFFER_LEN = 64;

typedef struct {
	//rendering
	float fov;
	float draw_distance;
	unsigned int max_depth;
	float bias;

	//window
	unsigned int width;
	unsigned int height;
	bool fullscreen;
	unsigned int fps;
}config;

static const config config_defaults = {
	//rendering
	90.0,
	200.0,
	5,
	0.1,
	
	//window
	2000,
	1000,
	false,
	120
};

char *get_file(const char *filename);
config parser(char *file_data);
config default_config(void);

#endif
