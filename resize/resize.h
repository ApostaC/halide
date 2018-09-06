#pragma once
#include <cstddef>

void resize_bilinear_halide(const float *src, float *dst,
        size_t src_h, size_t src_w, size_t src_step,
        size_t dst_h, size_t dst_w, size_t dst_step,
        size_t channels);

void resize_bilinear_naive(const float *src, float *dst,
        size_t src_h, size_t src_w, size_t src_step,
        size_t dst_h, size_t dst_w, size_t dst_step,
        size_t channels);
