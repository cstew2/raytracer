TARGET          := raytracer

MODULES         := main math world render debug

CC              := gcc
CUCC		:= gcc

CFLAGS          := -std=c11 -I.

CUFLAGS		:= 

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra
RCFLAGS         := -O2 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
#RCFLAGS         := -O2 -ftree-vectorize -finline-functions -s -march=native
PCFLAGS		:=

DCUFLAGS	:= -g -ggdb3 -O0 -Wall -pedantic -Wextra 
RCUFLAGS	:= -O2 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
#RCUFLAGS	:= -O2 -ftree-vectorize -finline-functions -s -march=native
PCUFLAGS	:= -pg

LIBS            := 

LDFLAGS		:= 
DLDFLAGS	:= 
RLDFLAGS	:= -flto -s
PLDFLAGS	:=

DCULDFLAGS	:= 
RCULDFLAGS	:= -flto -s
PCULDFLAGS	:=

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
debug: CUFLAGS += $(DCUFLAGS)
debug: LDFLAGS += $(DLDFLAGS)
debug: CULDFLAGS += $(DCULDFLAGS)
debug: build

.PHONY: release
release: CFLAGS += $(RCFLAGS)
debug: CUFLAGS += $(CURFLAGS)
debug: LDFLAGS += $(DLDFLAGS)
debug: CULDFLAGS += $(DCULDFLAGS)
release: build

.PHONY: profile
release: CFLAGS += $(PCFLAGS)
debug: CUFLAGS += $(PCUFLAGS)
debug: LDFLAGS += $(PLDFLAGS)
debug: CULDFLAGS += $(PCULDFLAGS)
release: build

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
