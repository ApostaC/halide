CXX = g++
CXX_FLAGS = -g -std=c++14 -fno-rtti -lpthread -lHalide -ldl

CUDA_ROOT = /usr/local/cuda-9.0
CXX_FLAGS += -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64

BASIC_FLAGS= -g -std=c++14 
ANDK_ROOT=/home/chengyihua/Application/android-ndk-r15c
ANDK_INC=${ANDK_ROOT}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/lib/gcc/aarch64-linux-android/4.9.x/include
CROSS_CXX=/home/chengyihua/Application/arm64_cross_api21/bin/aarch64-linux-android-clang++
CROSS_FLAG= ${BASIC_FLAGS} -static-libstdc++ -fPIE -Wl,-pie

#TARGET=host-cuda-cuda_capability_61
TARGET=host-opencl-debug

MATRIX_SIZE ?= 1024

all: bin/runner bin/mat_mul_arm

mat_mul.generator: mat_mul_gen.cpp GenGen.cpp
	${CXX} $^ -o $@ ${CXX_FLAGS}

bin/mat_mul.o: ./mat_mul.generator
	@mkdir -p bin
	./mat_mul.generator -g SimpleMMGen -o bin/ -e o,h,stmt \
		-f mat_mul target=${TARGET} size=${MATRIX_SIZE}

bin/mat_mul_arm.a: ./mat_mul.generator
	@mkdir -p bin
	./mat_mul.generator -g SimpleMMGen -o bin/ -e stmt,h,static_library \
		-f mat_mul -n mat_mul_arm target=arm-64-android-opencl size=${MATRIX_SIZE}

bin/mat_mul_arm.h: ./mat_mul.generator

bin/mat_mul_arm: runner.cpp bin/mat_mul_arm.h bin/mat_mul_arm.a 
	@mkdir -p bin
	${CROSS_CXX} ${CROSS_FLAG} $^ -O2 -llog -DARM
	scp a.out mi6@androidtest:chengyihua/bin/mat_mul
	touch mat_mul.generator

bin/runner: bin/mat_mul.o bin/mat_mul.h runner.cpp
	@mkdir -p bin
	${CXX} $^ -o $@ ${CXX_FLAGS} -lcudart -lcublas -O3 -DCMP_2_CUBLAS -DX86
	touch mat_mul.generator
