TARGET          := raytracer

MODULES         := main math world render debug

CC              := gcc
CUCC		:= nvcc

CFLAGS          := -std=c11 -I.

CUFLAGS		:= -std=c++11 -I.

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra
RCFLAGS         := -O3 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
PCFLAGS		:=

DCUFLAGS	:= -g -O0 
RCUFLAGS	:= -O3
PCUFLAGS	:= -pg

LD		:= gcc
CULD		:= nvcc

LDFLAGS		:=

DLDFLAGS	:= 
RLDFLAGS	:= -flto -s
PLDFLAGS	:=

DCULDFLAGS	:= 
RCULDFLAGS	:= 
PCULDFLAGS	:=

SRC		:= 
INC		:=

CUSRC		:= 
CUINC		:= 

THSRC		:= 
THINC		:= 

MPSRC		:=
MPINC		:=

GLSRC		:=
GLINC		:=

SDLSRC		:=
SDLINC		:=

VKSRC		:=
VKINC		:=

MPCFLAGS	:= 

LIBS            :=
CULIBS		:=
MPLIBS		:= 
GLLIBS		:=
SDLLIBS		:=
VKLIBS		:=

include         $(patsubst %, %/module.mk, $(MODULES))

VPATH		= $(MODULES)

PREFIX		:= build
BUILD 		?= debug

CUDA		?= 1
OPENMP		?= 1
PTHREAD		?= 1

OPENGL		?= 1
VULKAN		?= 1
SDL		?= 1

ifeq ($(BUILD), debug)
PREFIX	  	:= $(PREFIX)/debug
CFLAGS	  	+= $(DCFLAGS)
CUFLAGS   	+= $(DCUFLAGS)
LDFLAGS	  	+= $(DLDFLAGS)
LDFLAGS 	+= $(DCULDFLAGS)
else ifeq ($(BUILD),release)
PREFIX	  	:= $(PREFIX)/release
CFLAGS	  	+= $(RCFLAGS)
CUFLAGS   	+= $(RCUFLAGS)
LDFLAGS	  	+= $(RLDFLAGS)
LDFLAGS 	+= $(RCULDFLAGS)
else
$(error $(BUILD) is not a build profile)
endif

ifeq ($(CUDA), 1)
CUOBJ		:=$(patsubst %.cu,$(PREFIX)/obj/%.cu.o, $(filter %.cu,$(notdir $(CUSRC))))
LIBS		+= $(CULIBS)
LD		:= $(CULD)
LDFLAGS		:= $(CULDFLAGS)
endif

ifeq ($(OPENMP), 1)
SRC		+= $(MPSRC)
INC		+= $(MPINC)
CFLAGS		+= $(MPCFLAGS)
LIBS		+= $(MPLIBS)
endif

ifeq ($(PTHREAD), 1)
SRC		+= $(THSRC)
INC		+= $(THINC)
LIBS		+= $(THLIBS)
endif

ifeq ($(OPENGL), 1)
SRC		+= $(GLSRC)
INC		+= $(GLINC)
LIBS		+= $(GLLIBS)
endif

ifeq ($(VULKAN), 1)
SRC		+= $(VKSRC)
INC		+= $(VKINC)
LIBS		+= $(VKLIBS)
endif

ifeq ($(SDL), 1)
SRC		+= $(SDLSRC)
INC		+= $(SDLINC)
LIBS		+= $(SDLLIBS)
endif

.PHONY: release
release: $(PREFIX)/bin/$(TARGET)

$(shell mkdir -p $(PREFIX)/obj $(PREFIX)/bin)

OBJ		:=$(patsubst %.c, $(PREFIX)/obj/%.c.o,  $(filter %.c, $(notdir $(SRC))))

$(PREFIX)/bin/$(TARGET): $(OBJ) $(CUOBJ)
	@$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo [LD] Linked $^ into $@

$(PREFIX)/obj/%.c.o: %.c %.h
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo [CC] Compiled $< into $@

$(PREFIX)/obj/%.cu.o: %.cu %.cuh
	@$(CUCC) $(CUFLAGS) -c $< -o $@
	@echo [NVCC] Compiled $< into $@

.PHONY: rebuild
rebuild: clean $(PREFIX)/bin/$(TARGET)

.PHONY: install
install: $(PREFIX)/bin/$(TARGET)
	cp ./$(PREFIX)/bin/$(TARGET) ./

.PHONY: clean
clean:
	@rm -rf ./build core raytracer.log
	@echo Cleaned project


