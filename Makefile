TARGET          = raytracer

MODULES         = main math world render debug

CC              = gcc
CUCC		= gcc

CFLAGS          := -std=c11 -I.

CUFLAGS		:= 

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra -Wundef -Wshadow \
                   -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wwrite-strings \
                   -Wswitch-default -Wswitch-enum -Wunreachable-code -Winit-self \

RCFLAGS         := -O2 -fwhole-program -flto -ftree-vectorize -s -DNDEBUG -march=native


CUDFLAGS	:= -g
CURFLAGS	:= -O2
CUPFLAGS	:= -pg

LIBS            := 

LDFLAGS		:=
CULDFLAGS	:=

SRC		:= 
INC		:= 

CUSRC		:= 
CUINC		:= 


include         $(patsubst %, %/module.mk, $(MODULES))

OBJ		:= $(patsubst %.c,%.c.o, $(filter %.c,$(SRC)))
CUOBJ		:= $(patsubst %.cu,%.cu.o, $(filter %.cu,$(CUSRC)))

.PHONY: default
default:debug


.PHONY: debug 
debug: CFLAGS += $(DCFLAGS)
debug: CUFLAGS += $(CUDFLAGS)
debug: build

.PHONY: release
release: CFLAGS += $(RCFLAGS)
debug: CUFLAGS += $(CURFLAGS)
release: build

.PHONY: release
release: CFLAGS += $(PCFLAGS)
debug: CUFLAGS += $(CUPFLAGS)
release: build

.PHONY: cpu
cpu: CFLAGS += $(DCFLAGS)
cpu: cpubuild

cpubuild: $(OBJ)
	@$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET) $(LIBS)
	@echo [LD] Linked $^ into $(TARGET)

.PHONY: build
build: $(OBJ) $(CUOBJ)
	@$(CUCC) $(CULDFLAGS) $(OBJ) $(CUOBJ) -o $(TARGET) $(LIBS)
	@echo [LD] Linked $^ into $(TARGET)

%.c.o:%.c
	@$(CC) $(CFLAGS) -c $^ -o $@
	@echo [CC] Compiled $^ into $@

%.cu.o:%.cu
	@$(CUCC) $(CUFLAGS) -c $^ -o $@
	@echo [NVCC] Compiled $^ into $@

.PHONY: clean
clean:
	@rm -f $(OBJ) $(CUOBJ) $(TARGET) core raytracer.log
	@echo Cleaned $(OBJ) $(CUOBJ) and $(TARGET)

.PHONY: rebuild
rebuild: clean default
