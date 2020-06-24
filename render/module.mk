SRC		+= $(wildcard render/*.c)
INC		+= $(wildcard render/*.h)

SRC		+= render/ppm/render_file.c
INC		+= render/ppm/render_file.h

CCSRC		+= $(wildcard render/*.cc)
CCINC		+= $(wildcard render/*.cc)

GLSRC		+= render/opengl/gl_render.c
GLINC		+= render/opengl/gl_render.h

SDLSRC		+= render/sdl/sdl_render.c
SDLINC		+= render/sdl/sdl_render.h

VKSRC		+= render/vulkan/vk_render.c
VKINC		+= render/vulkan/vk_render.h

MPCFLAGS	+= -fopenmp

LIBS		+= -lm
CULIBS		+= -lcuda -lcudart
MPLIBS		+= -lgomp
THLIBS		+= -lpthread
GLLIBS		+= -lglfw -lGLEW -lGLU -lGL
SDLLIBS		+= -lSDL2
VKLIBS		+= -lglfw -lvulkan
