#include <cstddef>
#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include "resize.h"

template <typename T>
struct Mat {
    Mat(size_t h, size_t w, size_t c):
        h(h), w(w), c(c)
    {
        data.resize(h*w*c);
        fill_random();
    }
    void fill_random()
    {
        static std::mt19937 gen;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto &x: data) x = dist(gen);
    }
    T *ptr()
    { return data.data(); }
    size_t h, w, c;
    std::vector<T> data;
};

int main()
{
    size_t IH = 1080, IW = 1920, OH = 720, OW = 1280, C = 1;
    Mat<float> src(IH, IW, C), dst(OH, OW, C), dst_naive(OH, OW, C);
    // for (size_t i = 0; i < IH*IW*C; ++i) src.ptr()[i] = i;
    auto start = clock();
    for (size_t i = 0; i < 100; ++i) {
        resize_bilinear_halide(src.ptr(), dst.ptr(),
                src.h, src.w, src.w * src.c,
                dst.h, dst.w, dst.w * dst.c,
                C);
    }
    auto stop = clock();
    std::cout << "time is " << (float)(stop - start) / CLOCKS_PER_SEC * 1000 / 100 << std::endl;
    resize_bilinear_naive(src.ptr(), dst_naive.ptr(),
            src.h, src.w, src.w * src.c,
            dst.h, dst.w, dst.w * dst.c,
            C);
    for (size_t i = 0; i < OH*OW*C; ++i) {
        // printf("%f %f\n", dst_naive.ptr()[i], dst.ptr()[i]);
        assert(dst_naive.ptr()[i] == dst.ptr()[i]);
    }
}
