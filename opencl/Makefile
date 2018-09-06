CXX = g++
CXX_FLAGS = -g -std=c++11 -fno-rtti -lpthread -lHalide -ldl -lprofiler
HL_TARGET ?= host
GPU ?= false
MODE ?=NHWC

BASIC_FLAGS = -g -std=c++11
CROSS_CXX=/home/chengyihua/Application/arm64_cross_api21/bin/aarch64-linux-android-clang++
CROSS_FLAG= ${BASIC_FLAGS} -fPIE -Wl,-pie -static-libstdc++

SIZE ?= 1024

#====================================================================

all: bin/test  bin/test_arm

ocltest.generator: opencl_generator.cpp GenGen.cpp
	${CXX} $^ -o $@ ${CXX_FLAGS} 
bin/test_arm.a: ./ocltest.generator
	@mkdir -p bin
	./ocltest.generator -g OpenCLTest -o bin/ -e stmt,h,static_library	\
		-f Test -n test_arm target=arm-64-android-armv7s-opencl-debug size=${SIZE} 

bin/test_arm.h: ./ocltest.generator

bin/test_arm: test.cpp bin/test_arm.h bin/test_arm.a
	@mkdir -p bin
	${CROSS_CXX} ${CROSS_FLAG} $^ -llog -DSIZE=${SIZE} -DARM
	scp a.out mi6@androidtest:chengyihua/bin/opencl_test
	touch ocltest.generator


bin/test_x86.a:
	@mkdir -p bin
	./ocltest.generator -g OpenCLTest -o bin/ -e stmt,h,static_library	\
		-f Test -n test_x86 target=host-opencl-debug size=${SIZE}

bin/test_x86.h: ./ocltest.generator

bin/test: test.cpp bin/test_x86.h bin/test_x86.a
	@mkdir -p bin
	${CXX} ${CXX_FLAGS} $^ -DSIZE=${SIZE} -DX86 -o bin/test
