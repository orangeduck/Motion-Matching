# ==========================================
#  Motion-Matching - Linux (Ubuntu) Makefile
# ==========================================
PLATFORM := PLATFORM_DESKTOP
BUILD_MODE ?= RELEASE

# 头文件路径：raylib 和 raygui 安装到 /usr/local/
INCLUDE_DIR = -I./ -I/usr/local/include -I./raygui/src

# 库文件路径
LIBRARY_DIR = -L/usr/local/lib

# 预处理器定义
DEFINES = -D_DEFAULT_SOURCE -D_GNU_SOURCE -D$(PLATFORM)

# 编译器
CC = g++

# C++ 标准和基础编译选项
COMMON_CFLAGS = $(DEFINES) $(INCLUDE_DIR) -std=c++17

# 构建模式：Release 或 Debug
ifeq ($(BUILD_MODE), RELEASE)
    CFLAGS += $(COMMON_CFLAGS) -O3 -ffast-math -march=native -DNDEBUG
else
    CFLAGS += $(COMMON_CFLAGS) -g -O0
endif

# Linux 链接库（关键！替代 -lopengl32 等）
LIBS = -lraylib -lGL -ldl -lm -lpthread -lrt -lX11 \
       -lXrandr -lXinerama -lXi -lXcursor -lXxf86vm -lXext

# 源文件和头文件
SOURCE = $(wildcard *.cpp)
HEADER = $(wildcard *.h)

# 目标可执行文件名（Linux 不用加 .exe）
TARGET = controller

# 默认目标
.PHONY: all clean run

all: $(TARGET)

# 编译规则：单个 cpp 文件 → .o
%.o: %.cpp $(HEADER)
	$(CC) $(CFLAGS) -c $< -o $@

# 链接所有 .o 成可执行文件
$(TARGET): $(SOURCE:%.cpp=%.o)
	$(CC) -o $(TARGET) $^ $(LIBS)

# 清理中间文件和可执行文件
clean:
	rm -f *.o $(TARGET)

# 运行程序（便捷命令）
run: $(TARGET)
	./$(TARGET)
