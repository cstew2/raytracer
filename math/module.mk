SRC		+= $(wildcard math/*.c)
INC		+= $(wildcard math/*.h)

CCSRC		+= $(wildcard math/*.cc)
CCINC		+= $(wildcard math/*.cc)

CUSRC		+= $(wildcard math/*.cu)
CUINC		+= $(wildcard math/*.cuh)

LIBS		+= -lm
