cmake_minimum_required(VERSION 3.26)
project(libyolo)

set(CMAKE_CXX_STANDARD 17)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

#导入vulkan
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

set(OpenCV_DIR "./3rdpart/ocvm/x86/vc17/staticlib")
set(NCNN_INCLUDE_DIR "./3rdpart/ncnn/include")
include_directories(${NCNN_INCLUDE_DIR})
set(ncnn_DIR "./3rdpart/ncnn/lib/cmake/ncnn")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(./3rdpart/configor)

add_library(libyolo SHARED
        src/pch.cpp
        src/base_algorithm.cpp
        src/libyolo.cpp)

target_link_libraries(libyolo ncnn ${OpenCV_LIBS})
