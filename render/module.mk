SRC		+= $(wildcard render/*.c)
INC		+= $(wildcard render/*.h)

#CUSRC		+= $(wildcard render/*.cu)
#CUINC		+= $(wildcard render/*.cuh)

CFLAGS		+= -I/usr/include/SDL2
LIBS		+= -lglfw -lGLEW -lGLU -lGL -lvulkan -lSDL2 -lm
