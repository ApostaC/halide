#include "./resize_f32.h"
#include "./resize_u8.h"
#include "./resizewm.h"

#include <HalideBuffer.h>
#include <ctime>
void test_f32()
{
    Halide::Runtime::Buffer<float> input(1080, 1920, 1), output(720, 1280, 1);
    input.fill(1.0f);
    auto run = [&]() {
        resize_f32(input, 2.0f/3.0f, output);
    };
    run();
    auto start = clock();
    for (size_t i = 0; i < 100; ++i) {
        run();
    }
    auto stop = clock();
    printf("1920x1080 -> 1280x720 f32 halide time is %fms\n", (float)(stop - start) / CLOCKS_PER_SEC * 1000 / 100);
}

void test_u8()
{
    Halide::Runtime::Buffer<float> input(1080, 1920, 1), output(720, 1280, 1);
    input.fill(1.0f);
    auto run = [&]() {
        resize_u8(input, 2.0f/3.0f, output);
    };
    run();
    auto start = clock();
    for (size_t i = 0; i < 100; ++i) {
        run();
    }
    auto stop = clock();
    printf("1920x1080 -> 1280x720 u8 halide time is %fms\n", (float)(stop - start) / CLOCKS_PER_SEC * 1000 / 100);
}

void test_wm()
{
    Halide::Runtime::Buffer<float> input(1080, 1920, 1), output(720, 1280, 1);
    auto ptr = input.data();
    for (size_t i = 0; i < input.number_of_elements(); ++i) {
        ptr[i] = (float)rand() / RAND_MAX;
    }
    auto run = [&]() {
        input.set_host_dirty();
        resizewm(input, 720, 1280, output);
        output.device_sync();
        output.copy_to_host();
    };
    run();
    auto start = clock();
    for (size_t i = 0; i < 100; ++i) {
        run();
    }
    auto stop = clock();
    for (size_t i = 0; i < 5; ++i) {
        printf("%f %f ", input.data()[i], output.data()[i]);
    }
    printf("\n");
    printf("1920x1080 -> 1280x720 f32 halide time is %fms\n", (float)(stop - start) / CLOCKS_PER_SEC * 1000 / 100);
}

int main()
{
    test_f32();
    test_wm();
}
