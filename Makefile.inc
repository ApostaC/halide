# Makefile.inc
# defines some basic env variables
CXX = g++
CXX_FLAGS = -g -std=c++14 -fno-rtti -lpthread -lHalide -ldl

CUDA_ROOT = /usr/local/cuda-9.0
CXX_FLAGS += -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64

BASIC_FLAGS= -g -std=c++14 
CROSS_CXX=/home/chengyihua/Application/arm64_cross_api21/bin/aarch64-linux-android-clang++
CROSS_FLAG= ${BASIC_FLAGS} -static-libstdc++ -fPIE -Wl,-pie
