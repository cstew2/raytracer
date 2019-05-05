#ifndef __RENDER_FILE_H__
#define __RENDER_FILE_H__

#include "main/config.h"
#include "render/raytracer.h"

void file_render(raytracer rt);
void write_ppm_file(char *filename, canvas c);

#endif

