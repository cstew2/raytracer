#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "fb_render.h"

#include "debug/debug.h"

void fb_realtime_render(raytracer rt)
{
	log_msg(INFO, "Starting main game loop\n");
	while(0) {
		fb_input();
		fb_update();
		fb_render();
	}
	fb_cleanup();
}

int fb_init(config c)
{
	int fb_fd = open(DEV_FB, O_WRONLY);
	if(fb_fd == -1) {
		log_msg(ERROR, "");
		return -1;
	}

	struct stat sb;
	int fb_size = fstat(fb_fd, &sb);
	if (fb_size == -1) {
		close(fb_fd);
	        log_msg(ERROR, "");
		return -1;
	}
	
	char *fb_addr = mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
	if(fb_addr == MAP_FAILED) {
		log_msg(ERROR, "");
		return -1;
	}

	
	munmap(fb_addr, fb_size);
	close(fb_fd);

	return 0;
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

