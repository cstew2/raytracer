#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <pthread.h>
#include <unistd.h>

#include "render/threaded_raytracer.h"
#include "render/cpu_raytracer.h"

#define THREAD_COUNT 8
pthread_t threads[THREAD_COUNT];
threaded_args args[THREAD_COUNT];

int threaded_render(const raytracer rt)
{
	int work = (rt.camera.width*rt.camera.height)/THREAD_COUNT;
	
	for(int t=0; t < THREAD_COUNT; t++) {
		args[t].rt = rt;
		args[t].c = calloc(sizeof(vec4), work);
		args[t].id = t;
		pthread_create(&threads[t], NULL, &threaded_render_work,
			       (void*) &args[t]);
	}

	for(int t=0; t < THREAD_COUNT; t++) {
		pthread_join(threads[t], NULL);
		for(int i=t; i < work; i++) {
			int x = ((work*t) + i) % rt.camera.width;
			int y = ((work*t) + i) / rt.camera.width;
			canvas_set_pixel(rt.canvas, x, y, args[t].c[i]);
		}
		free(args[t].c);
	}
	
	return 0;
}

void *threaded_render_work(void *args)
{
	threaded_args *ta = (threaded_args *)args;
	int work = ((ta->rt.camera.width*ta->rt.camera.height)/THREAD_COUNT);
        int start = work * ta->id;
	int end = work * (ta->id+1);

	for(int i=0; i < work; i++) {
		int x = (start + i) % ta->rt.camera.width;
		int y = (start + i) / ta->rt.camera.width;
		ray r = generate_ray(ta->rt.camera, x, y);
		ta->c[i] = cpu_cast_ray(r, ta->rt);
	}
	
	return NULL;
}
