#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"


int main(int argc, char **argv)
{
	if(argc > 1) {
		char *filename;
		for(int i=0; i < argc; i++) {
			if(strncmp(argv[i], "-f", 2)) {
				i++;
				filename = malloc(sizeof(char) * strlen(argv[i] + 1));
				strncpy(filename, argv[i], strlen(argv[i] + 1));
			}
		}
	}
	
	return 0;
}
