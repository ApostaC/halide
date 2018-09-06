CXX = g++
CXX_FLAGS = -g -std=c++11 -fno-rtti -lpthread -lHalide -ldl -lprofiler
HL_TARGET ?= host
GPU ?= false
MODE ?=NHWC

BASIC_FLAGS = -g -std=c++11
CROSS_CXX=/home/chengyihua/Application/arm64_cross_api21/bin/aarch64-linux-android-clang++
CROSS_FLAG= ${BASIC_FLAGS} -fPIE -Wl,-pie -static-libstdc++

all: bin/Convolution bin/Convolution_arm 

convolution.generator: Convolution_gen.cpp GenGen.cpp
	${CXX} $^ -o $@ ${CXX_FLAGS}

bin/Convolution.a: ./convolution.generator
	@mkdir -p bin
	./convolution.generator -g Convolution_${MODE} -o bin/ -e stmt,h,static_library \
		-f Convolution target=${HL_TARGET} usingGPU=${GPU}

bin/Convolution.h: convolution.generator

bin/Convolution: Convolution.cpp bin/Convolution.h bin/Convolution.a 
	@mkdir -p bin
	${CXX} $^ -o $@ ${CXX_FLAGS} -O2 -D${MODE} #-DUSE_PPROF
	touch convolution.generator

bin/Convolution_arm.a: ./convolution.generator
	@mkdir -p bin
	./convolution.generator -g Convolution_${MODE} -o bin/  -e stmt,h,static_library \
		-f Convolution -n Convolution_arm target=arm-64-android-armv7s-opencl-debug

bin/Convolution_arm.h: convolution.generator

bin/Convolution_arm: Convolution.cpp bin/Convolution_arm.h bin/Convolution_arm.a 
	@mkdir -p bin
	${CROSS_CXX} ${CROSS_FLAG} $^ -O2 -llog -D${MODE}
	scp a.out mi6@androidtest:chengyihua/conv/${MODE}
	touch convolution.generator

