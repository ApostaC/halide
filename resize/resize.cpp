#include "resize.h"

#include <Halide.h>

void resize_bilinear_halide(const float *src, float *dst,
        size_t src_h, size_t src_w, size_t src_step,
        size_t dst_h, size_t dst_w, size_t dst_step,
        size_t channels)
{
    Halide::Runtime::Buffer<float> input_buffer(
            const_cast<float *>(src),
            {static_cast<int>(src_w),
            static_cast<int>(src_h),
            static_cast<int>(channels)});
    Halide::Runtime::Buffer<float> output_buffer(
            const_cast<float *>(dst),
            {static_cast<int>(dst_w),
            static_cast<int>(dst_h),
            static_cast<int>(channels)});

    Halide::Buffer<float> input(std::move(input_buffer), "input");
    Halide::Buffer<float> output(std::move(output_buffer), "output");
    float coefh = static_cast<float>(src_h) / dst_h;
    float coefw = static_cast<float>(src_w) / dst_w;
    Halide::Var oh, ow, c, y, x;
    Halide::Expr ih = oh * coefh, iw = ow * coefw;
    Halide::Expr ih_s = Halide::cast<int>(Halide::floor(ih)), iw_s =
        Halide::cast<int>(Halide::floor(iw));
    Halide::Expr ih_f = ih - ih_s, iw_f = iw - iw_s;
    Halide::Func input_clamped = Halide::BoundaryConditions::repeat_edge(input);
    Halide::Func result;
    Halide::Func phase1;
    phase1(ow, y, c) =
        input_clamped(iw_s, y, c)*(1.0f - iw_f) +
        input_clamped(iw_s+1, y, c)*iw_f;
    result(ow, oh, c) =
        phase1(ow, ih_s, c)*(1.0f - ih_f)+
        phase1(ow, ih_s+1, c)*ih_f;
    result.realize(output);
}

void resize_bilinear_naive(const float *src, float *dst,
        size_t src_h, size_t src_w, size_t src_step,
        size_t dst_h, size_t dst_w, size_t dst_step,
        size_t channels)
{
    float coefh = static_cast<float>(src_h) / dst_h;
    float coefw = static_cast<float>(src_w) / dst_w;
    for (size_t oh = 0; oh < dst_h; ++oh) for (size_t ow = 0; ow < dst_w; ++ow)
    {
        float ih = oh*coefh, iw = ow*coefw;
        size_t ih_s = std::floor(ih), iw_s = std::floor(iw);
        float ih_f = ih - ih_s, iw_f = iw - iw_s;
        auto clamp = [](size_t x, size_t X) {
            if (x >= X) {
                return X - 1;
            } else {
                return x;
            }
        };
        size_t h0 = clamp(ih_s, src_h), h1 = clamp(ih_s+1, src_h);
        size_t w0 = clamp(iw_s, src_w), w1 = clamp(iw_s+1, src_w);
        for (size_t c = 0; c < channels; ++c) {
            dst[ow + oh*dst_w + c*dst_h*dst_w] =
                src[w0 + h0*src_w + c*src_h*src_w]*(1.0f-ih_f)*(1.0f-iw_f) +
                src[w1 + h0*src_w + c*src_h*src_w]*(1.0f-ih_f)*iw_f +
                src[w0 + h1*src_w + c*src_h*src_w]*ih_f*(1.0f-iw_f) +
                src[w1 + h1*src_w + c*src_h*src_w]*ih_f*iw_f;
        }
    }
}
