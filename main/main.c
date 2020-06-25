#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "main/main.h"
#include "debug/debug.h"

#include "render/raytracer.h"

//rendering backends
#if USE_OPENGL
#include "render/opengl/gl_render.h"
#endif

#if USE_VULKAN
#include "render/vulkan/vk_render.h"
#endif

#if USE_SDL
#include "render/sdl/sdl_render.h"
#endif

#include "render/ppm/render_file.h"

//computation backends
#include "compute/cpu/cpu_raytracer.h"

#if USE_PTHREAD
#include "compute/pthread/pthread_raytracer.h"
#endif

#if USE_OPENMP
#include "compute/openmp/openmp_raytracer.h"
#endif

#if USE_CUDA
#include "compute/cuda/cuda_raytracer.cuh"
#endif

#include "main/config.h"

int main(int argc, char **argv)
{
	log_init(ERROR);
	char *filename = NULL;
	char *config_path = NULL;
	config c;

	bool picture = false;
	bool opengl = false;
	bool vulkan = false;
	bool sdl = false;

	bool threads = false;
	bool openmp = false;
	bool cuda = false;
	
	//check args
	if(argc > 1) {
		for(int i=0; i < argc; i++) {
			if(!strncmp(argv[i], "-h", 2)) {
				print_help();
				return 0;
			}
			if(!strncmp(argv[i], "-d", 2)) {
				set_log_level(DEBUG);
			}
			else if(!strncmp(argv[i], "-f", 2)) {
				i++;
				if(argv[i]) {
					filename = malloc(sizeof(char) * (strlen(argv[i]) + 1));
					strcpy(filename, argv[i]);
				}
				else {
					log_msg(WARN, "You passed the -f flag without any parameter" \
						"after it, it will be ignored\n");
				}
			}
			else if(!strncmp(argv[i], "-c", 2)) {
				i++;
				log_msg(INFO, "Loading configuration file: %s\n", argv[i]);
				config_path = malloc(sizeof(char) * (strlen(argv[i]) + 1));
				strcpy(config_path, argv[i]);
			}
			else if(!strncmp(argv[i], "--ppm", 5)) {
				picture = true;
			}
			else if(!strncmp(argv[i], "--gl", 4)) {
				opengl = true;
			}
			else if(!strncmp(argv[i], "--vk", 4)) {
				vulkan = true;
			}
			else if(!strncmp(argv[i], "--sdl", 5)) {
				sdl = true;
			}
			else if(!strncmp(argv[i], "--cuda", 6)) {
				cuda = true;
			}
			else if(!strncmp(argv[i], "--openmp", 8)) {
				openmp = true;
			}
			else if(!strncmp(argv[i], "--pthread", 9)) {
				threads = true;
			}
		}
	}

	//decide what to do with args 
	if(config_path != NULL) {
		char *f = get_file(config_path);
		c = parser(f);
		free(f);	      
	}
	else {
		c = default_config();
	}

	//do this for all command line options
	if(opengl) {
		c.render_method = OPENGL;
	}

	set_log_level(c.log_level);
	
	raytracer rt = raytracer_test(c);

	int (*compute)(raytracer, void *) = &cpu_render;
		
	if(threads) {
		#ifdef USE_PTHREAD
		compute = &pthread_render;
		#else
		log_msg(ERROR, "You need to compile with pthread support to use --pthread\n");
	        return -1;
		#endif
	}
	if(openmp) {
		#ifdef USE_OPENMP
		compute = &openmp_render;
		#else
		log_msg(ERROR, "You need to compile with OPENMP support to use --openmp\n");
		return -1;
		#endif
	}
	if(cuda) {
		#ifdef USE_CUDA
		compute = &cuda_render;
		#else
		log_msg(ERROR, "You need to compile with CUDA support to use --cuda\n");
		return -1;
		#endif	
	}
	
	//change this to use config
	//refactor entry-point for render_method and raytracer method
	if(picture && filename != NULL) {
		log_msg(INFO, "Writing to:%s\n", filename);
		file_render(rt, filename);
	}
	else {
		if(opengl && vulkan && sdl) {
			log_msg(WARN, "You passed too many arguments for the render\n");
		}
		else if(opengl && !vulkan && !sdl) {
			#ifdef USE_OPENGL
		        gl_realtime_render(rt, compute);
                        #else
			log_msg(ERROR, "You did not compile with OpenGL support\n");
			#endif 
		}
		else if(!opengl && vulkan && !sdl) {
			#ifdef USE_VULKAN
		        vk_realtime_render(rt, compute);
			#else
			log_msg(ERROR, "You did not compile with Vulkan support\n");
			#endif 
		}
		else if(!opengl && !vulkan && sdl) {
			#ifdef USE_SDL
			sdl_realtime_render(rt, compute);
			#else
			log_msg(ERROR, "You did not compile with SDL support\n");
			#endif 
		}
		else if(!(opengl && vulkan && sdl)) {
			#ifdef USE_OPENGL
			log_msg(WARN, "No argument selected for the renderer, defaulting to opengl\n");
			gl_realtime_render(rt, compute);
			#else
			log_msg(ERROR, "You did not compile with any renderer support\n");
			#endif 
		}
	}
	
	raytracer_term(rt);
	log_term();
	
	if(filename != NULL) {
		free(filename);
	}
	if(config_path != NULL) {
		free(config_path);
	}
	
	return 0;
}

void print_help(void)
{
	printf("cudaray usage: cudaray [OPTIONS]...\n\n"
	       "Options\n"
	       "-d \t debug on\n"
	       "-c \t pass a config file\n"
	       "-f \t open \"FILE\"\n"
	       "-gl \t use opengl to render\n"
	       "-vk \t use vulkan to render\n"
	       "-sdl \t use SDL to render\n"
	       "-p \t render single image to pnm file\n"
	       "-cuda \t use cuda to render "
	       "-openmp \t use openmp to render"
	       "-threads \t use pthreads to render");
}
