#include <iostream>
#include "Halide.h"

int main()
{
    Halide::Target target = Halide::get_host_target();
    target = Halide::get_target_from_environment();
    target.set_feature(Halide::Target::CUDA);
    target.set_feature(Halide::Target::OpenCL);
    bool gpu = target.has_gpu_feature();
    std::cout<<gpu<<std::endl;
}

