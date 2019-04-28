SRC		+= $(wildcard math/*.c)
INC		+= $(wildcard math/*.h)

CUSRC		+= $(wildcard math/*.cu)
CUINC		+= $(wildcard math/*.cuh)

LIBS		+= -lm
