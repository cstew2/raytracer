TARGET          = raytracer

MODULES         = main math world render voxel debug

CC              = gcc

CFLAGS          := -std=c11 -I./

DCFLAGS         := -g -ggdb3 -O0 -Wall -pedantic -Wextra -Wundef -Wshadow \
                   -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wwrite-strings \
                   -Wswitch-default -Wswitch-enum -Wunreachable-code -Winit-self \
		   -Werror -Wuninitialized

RCFLAGS         := -O2 -fwhole-program -s -DNDEBUG 

LIBS            := 

include         $(patsubst %, %/module.mk, $(MODULES))

OBJ             := $(patsubst %.c,%.o, $(filter %.c,$(SRC)))


.PHONY: debug
debug: CFLAGS += $(DCFLAGS)
debug: build

.PHONY: release
release: CFLAGS += $(RCFLAGS)
release: build

.PHONY: build
build: $(OBJ)
	@$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET) $(LIBS)
	@echo [LD] Linked $^ into $(TARGET)

%.o:%.c
	@$(CC) $(CFLAGS) -c $^ -o $@
	@echo [CC] Compiled $^ into $@

.PHONY: clean
clean:
	@rm -f $(OBJ) $(TARGET) core raytracer.log
	@echo Cleaned $(OBJ) and $(TARGET)

.PHONY: rebuild
rebuild: clean build
