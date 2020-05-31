TARGET          := raytracer

MODULES         := main math world render debug

CC              := gcc
CCCC		:=
CUCC		:= nvcc

CFLAGS          := -std=c11 -I. -x c
CCCFLAGS	:= -x cu 
CUFLAGS		:= -I. --std=c++11 -arch=sm_61 -dc

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra
RCFLAGS         := -O3 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
PCFLAGS		:=

DCUFLAGS	:= -g -G -O0
RCUFLAGS	:= -O3
PCUFLAGS	:= -pg

LD		:= gcc
CULD		:= nvcc

LDFLAGS		:= -L/opt/cuda/lib64
CULDFLAGS	:= -arch=sm_61 -dlink

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

LIBS            := -lstdc++
CULIBS		:= -dlink
MPLIBS		:= 
GLLIBS		:=
SDLLIBS		:=
VKLIBS		:=

VPATH		= $(MODULES)

PREFIX		:= build
BUILD 		?= debug

CUDA		?= 1
OPENMP		?= 1
PTHREAD		?= 1

OPENGL		?= 1
VULKAN		?= 1
SDL		?= 1

include         $(patsubst %, %/module.mk, $(MODULES))

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
LIBS		+= $(CULIBS)
CCCC		:= $(CUCC)
CCCFLAGS	+= $(CUFLAGS)
LDFLAGS		+=
CUOBJ		:= $(patsubst %.cu,$(PREFIX)/obj/%.cu.o, $(filter %.cu,$(notdir $(CUSRC))))
CUTARGET	:= $(PREFIX)/obj/cuda_code.cuo
else
CCCC		:= $(CC)
CCCFLAGS	:= $(CFLAGS)
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

OBJ		:= $(patsubst %.c, $(PREFIX)/obj/%.c.o,  $(filter %.c, $(notdir $(SRC))))
CCOBJ		:= $(patsubst %.cc,$(PREFIX)/obj/%.cc.o, $(filter %.cc,$(notdir $(CCSRC))))

$(PREFIX)/bin/$(TARGET): $(OBJ) $(CCOBJ) $(CUOBJ) $(CUTARGET)
	@$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo [LD] Linked $^ into $@

$(PREFIX)/obj/%.c.o: %.c %.h
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo [CC] Compiled $< into $@

$(PREFIX)/obj/%.cu.o: %.cu %.cuh
	@$(CUCC) $(CUFLAGS) -c $< -o $@
	@echo [NVCC] Compiled $< into $@

$(PREFIX)/obj/%.cc.o: %.cc %.hh
	@$(CCCC) $(CCCFLAGS) -c $< -o $@
	@echo [CCCC] Compiled $< into $@

$(CUTARGET): $(CCOBJ) $(CUOBJ)
	@$(CULD) $(CULDFLAGS) $^ -o $@ $(CULIBS)
	@echo [CULD] Linked $^ into $@

.PHONY: rebuild
rebuild: clean $(PREFIX)/bin/$(TARGET)

.PHONY: install
install: $(PREFIX)/bin/$(TARGET)
	cp ./$(PREFIX)/bin/$(TARGET) ./

.PHONY: clean
clean:
	@rm -rf ./build core raytracer.log
	@echo Cleaned project


