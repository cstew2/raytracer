TARGET          := raytracer

MODULES         := main math world render debug

CC              := gcc
CUCC		:= nvcc

CFLAGS          := -std=c11 -I.

CUFLAGS		:= -std=c++11 -I.

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra
RCFLAGS         := -O3 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
RCFLAGS         := -O3 -ftree-vectorize -finline-functions -s -march=native
PCFLAGS		:=

DCUFLAGS	:= -g -O0 
RCUFLAGS	:= -O3 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -march=native
RCUFLAGS	:= -O3 -ftree-vectorize -finline-functions -s -march=native
PCUFLAGS	:= -pg

LD		:= gcc
CULD		:= nvcc

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

LIBS            := 

include         $(patsubst %, %/module.mk, $(MODULES))

VPATH		= $(MODULES)

PREFIX		:= build
BUILD 		:= debug


ifeq ($(BUILD),debug)
PREFIX	  := $(PREFIX)/debug
CFLAGS	  += $(DCFLAGS)
LDFLAGS	  += $(DLDFLAGS)
CUFLAGS   += $(DCUFLAGS)
CULDFLAGS += $(DCULDFLAGS)
else ifeq ($(BUILD),release)
PREFIX	  := $(PREFIX)/release
CFLAGS	  += $(RCFLAGS)
LDFLAGS	  += $(RLDFLAGS)
CUFLAGS   += $(RCUFLAGS)
CULDFLAGS += $(RCULDFLAGS)
else
$(error BUILD not found)
endif

$(shell mkdir -p $(PREFIX)/obj $(PREFIX)/bin)

OBJ		:=$(patsubst %.c, $(PREFIX)/obj/%.c.o,  $(filter %.c, $(notdir $(SRC))))
CUOBJ		:=$(patsubst %.cu,$(PREFIX)/obj/%.cu.o, $(filter %.cu,$(notdir $(CUSRC))))

$(PREFIX)/bin/$(TARGET): $(OBJ) $(CUOBJ)
	@$(CULD) $(CULDFLAGS) $^ -o $@ $(LIBS)
	@echo [LD] Linked $^ into $@

$(PREFIX)/obj/%.c.o: %.c %.h
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo [CC] Compiled $< into $@

$(PREFIX)/obj/%.cu.o: %.cu %.cuh
	$(CUCC) $(CUFLAGS) -c $< -o $@
	@echo [NVCC] Compiled $< into $@

.PHONY: debug
debug: BUILD := debug
debug: $(PREFIX)/bin/$(TARGET)

.phony: release
release: BUILD := release
release: $(PREFIX)/bin/$(TARGET)

.PHONY: rebuild
rebuild: clean $(PREFIX)/bin/$(TARGET)

.PHONY: run
run:
	./$(PREFIX)/bin/$(TARGET)

.PHONY: clean
clean:
	@rm -f $(OBJ) $(CUOBJ) $(TARGET) core raytracer.log
	@echo Cleaned $(OBJ) $(CUOBJ) and $(TARGET)


