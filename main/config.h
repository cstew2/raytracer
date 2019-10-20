#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdbool.h>

static const int BUFFER_LEN = 64;
static const char COMMENT_PREFIX = '#';
static const char ASSIGNMENT_SEPERATOR = '=';

typedef enum CONFIG_TYPES = {
	BOOL,
	CHAR,
	STRING,
	INT,
	FLOAT
};

typedef enum RAYTRACE_METHOD {
	CPU,
	MULTITHREADED,
	CUDA,
	OPENCL
};

typedef enum RENDER_METHOD {
	FILE,
	OPENGL,
	VULKAN,
	SDL,
	LINUX_FB
};

typedef struct {
	//platform rendering
	RAYTRACE_METHOD raytrace_method;
	RENDER_METHOD render_method;
	
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
	//platform rendering
	CPU,
	OPENGL,
	
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
