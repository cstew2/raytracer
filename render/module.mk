SRC		+= $(filter-out render/gl_render.c \
				render/sdl_render.c \
				render/vk_render.c \
				render/openmp_raytracer.c \
				render/threaded_raytracer.c, \
		                $(wildcard render/*.c))

INC		+= $(filter-out render/gl_render.h \
				render/sdl_render.h \
				render/vk_render.h \
				render/openmp_raytracer.h \
				render/threaded_raytracer.h, \
		                $(wildcard render/*.h))

CCSRC		+= $(wildcard render/*.cc)
CCINC		+= $(wildcard render/*.cc)

CUSRC		+= $(wildcard render/*.cu)
CUINC		+= $(wildcard render/*.cuh)

MPSRC		+= openmp_raytracer.c
MPINC		+= openmp_raytracer.h

THSRC		+= threaded_raytracer.c
THINC		+= threaded_raytracer.h

GLSRC		+= gl_render.c
GLINC		+= gl_render.h

SDLSRC		+= sdl_render.c
SDLINC		+= sdl_render.h

VKSRC		+= vk_render.c
VKINC		+= vk_render.h

MPCFLAGS	+= -fopenmp

LIBS		+= -lm
CULIBS		+= -lcuda -lcudart
MPLIBS		+= -lgomp
THLIBS		+= -lpthread
GLLIBS		+= -lglfw -lGLEW -lGLU -lGL
SDLLIBS		+= -lSDL2
VKLIBS		+= -lglfw -lvulkan
