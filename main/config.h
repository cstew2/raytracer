#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdbool.h>

static const int BUFFER_LEN = 64;
static const char COMMENT_PREFIX = '#';
static const char ASSIGNMENT_SEPERATOR = '=';

typedef enum {
	BOOL,
	CHAR,
	STRING,
	INT,
	FLOAT
}CONFIG_TYPES;

typedef enum {
	CPU,
	MULTITHREADED,
	CUDA,
	OPENCL
}RAYTRACE_METHOD;

typedef enum {
	PPM,
	OPENGL,
	VULKAN,
	SDL,
	LINUX_FB
}RENDER_METHOD;

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

char *get_file(const char *filename);
config parser(char *file_data);
config default_config(void);

#endif
