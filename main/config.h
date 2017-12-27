#ifndef _CONFIG_H_
#define _CONFIG_H_

#define BUFFER_LEN 64
#define CONFIG_COUNT 3

typedef enum {INT, FLOAT, STRING} type;

typedef struct {
	char name[BUFFER_LEN];
	type t;
	union {
		int i;
		float f;
		char *s;
	};
}config_element;

static const config_element config_defaults[CONFIG_COUNT] = {
	{"fov", INT, {0}},
	{"draw_distance", FLOAT, {200.0}},
	{"", INT, {0}}
};

char *get_file(const char *filename);
config_element *parser(char *file_data);



#endif
