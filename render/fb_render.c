#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "fb_render.h"

void fb_realtime_render(raytracer rt)
{
	
}

void fb_init(config c)
{
	fb_dev = open(DEV_FB, 0_WRONLY);
	if(fb_dev == -1) {
		log_msg();
	}
	char *addr;
	addr = mmap(NULL, 
}

void fb_render(void)
{
	
}

void fb_input(void)
{
	
}

void fb_update(void)
{
	
}

void fb_cleanup(void)
{
	
}

