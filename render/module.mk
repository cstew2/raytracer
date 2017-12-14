SRC		+= $(wildcard render/*.c)
INC		+= $(wildcard render/*.h)

GLLIBS		+= -lglfw -lGLEW -lGLU -lGL
VKLIBS		+= -lglfw
