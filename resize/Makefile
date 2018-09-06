CXX = clang++
CXXFLAGS = -g -I/home/chengyihua/.local/include -I/home/chengyihua/Application/Halide-src \
		   -std=c++14

ANDK_ROOT=/home/chengyihua/Application/android-ndk-r15c
ANDK_INC=${ANDK_ROOT}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/lib/gcc/aarch64-linux-android/4.9.x/include

CXX_CROSS = /home/chengyihua/Application/arm64_cross_api21/bin/aarch64-linux-android-clang++
CXXFLAGS_CROSS = -I${ANDK_INC} $(CXXFLAGS) -fPIE -Wl,-pie

all: resize_expr

%.o: %.cpp *.h
	$(CXX) $< -o $@ $(CXXFLAGS) -c

resize_generator: resize_generator.cpp
	$(CXX) $^ ${CXXFLAGS} ~/Application/Halide-src/tools/GenGen.cpp -o $@ -Wl,-rpath=~/.local/lib -lHalide -fno-rtti

resizewm.a: resize_generator
	./resize_generator -g resizewm target=arm-64-android-opencl -o . input.type=float32 -f resizewm

resizewm.h: resize_generator

resize_f32.a: resize_generator
	./resize_generator -g resize target=arm-64-android -o . input.type=float32 -f resize_f32

resize_f32.h: resize_generator

resize_u8.a: resize_generator
	./resize_generator -g resize target=arm-64-android -o . input.type=uint8 -f resize_u8

resize_u8.h: resize_generator

resize_expr: resize_expr.cpp resize_u8.h resize_f32.h resize_u8.a resize_f32.a resizewm.h resizewm.a
	$(CXX_CROSS) resize_expr.cpp resize_u8.a resize_f32.a resizewm.a -o $@ $(CXXFLAGS_CROSS) -llog

deploy: resize_expr
	scp resize_expr mi5s@androidtest:~/weiming

run: deploy
	ssh mi5s@androidtest '~/weiming/resize_expr'

clean:
	rm *.o a.out resize_generator resize_u8.h resize_u8.a resize_f32.h resize_f32.a

.PHONY: all clean deploy run
