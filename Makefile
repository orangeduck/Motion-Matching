CC = g++

RAYLIB_DIR = C:/raylib/
INCLUDE_DIR = -I ./ -I $(RAYLIB_DIR)/raylib/src -I $(RAYLIB_DIR)/raygui/src
LIBRARY_DIR = -L $(RAYLIB_DIR)/raylib/src
DEFINES = -D RAYLIB_BUILD_MODE=RELEASE
CFLAGS ?= -O3 $(INCLUDE_DIR) $(LIBRARY_DIR) $(DEFINES)

LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm

SOURCE = $(wildcard *.c)
EXT = .exe

.PHONY: all

all: controller

controller: $(SOURCE) $(HEADER)
	$(CC) $(CFLAGS) $(SOURCE) -o $@$(EXT) $(LIBS) 

clean:
	rm controller$(EXT)