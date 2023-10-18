cmake_minimum_required(VERSION 3.15)

project(controller)

find_package(raylib 3.0 REQUIRED)

# Download raygui.h if it doesn't exist
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/raygui.h)
  file(DOWNLOAD
    https://raw.githubusercontent.com/raysan5/raygui/master/src/raygui.h
    ${CMAKE_SOURCE_DIR}/raygui.h)
endif()

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} controller.cpp)

target_link_libraries(${PROJECT_NAME} raylib)

# Checks if OSX and links appropriate frameworks (only required by raylib on MacOS)
if(APPLE)
  target_link_libraries(${PROJECT_NAME} "-framework IOKit")
  target_link_libraries(${PROJECT_NAME} "-framework Cocoa")
  target_link_libraries(${PROJECT_NAME} "-framework OpenGL")
endif()
