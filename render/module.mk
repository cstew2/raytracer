SRC		+= $(wildcard render/*.c)
INC		+= $(wildcard render/*.h)

LIBS		+= -lglfw -lGLEW -lGLU -lGL -lvulkan
