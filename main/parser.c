#include <stdio.h>
#include <stdlib.h>

#include "parser.h"

char *get_file(char *filename)
{
	FILE *f = NULL;
	size_t size;
	f = fopen(filename, "rb");
	
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	
	char *file_buffer = calloc(size, sizeof(char));
	fread(file_buffer, 1, size, f);
	fclose(f);

	return file_buffer;
}
