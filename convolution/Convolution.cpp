#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <cmath>
#include <limits>
#include <random>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/fcntl.h>

#include "bin/Convolution.h"
#include "bin/Convolution_arm.h"
#include "halide_benchmark.h"

#include "HalideBuffer.h"

#ifdef USE_PPROF 
#include <gperftools/profiler.h>
#endif
typedef float Data_t;

int profile(int oc, int ic, int w, int h, int kernel, int strd, int runs = 50)
{
//    if(argc <= 8)
//    {
//        fprintf(stderr, "Usage: %s N H W C FH FW OC [stride]\n", argv[0]);
//        return -1;
//    }

    int N = 1;//atoi(argv[1]);
    int H = h;//atoi(argv[2]);
    int W = w;//atoi(argv[3]);
    int C = ic;//atoi(argv[4]);
    int FH = kernel;// atoi(argv[5]);
    int FW = kernel;// atoi(argv[6]);
    int OC = oc;// atoi(argv[7]);
    int stride = strd;//1;
    //if(argc > 8) stide = atoi(argv[8]);

    printf("Benchmarking %dx%dx%dx%d\n", N, H, W, C);
    printf("\tFilter is %dx%d\n",FH,FW);
    printf("\tOutput Channel is %d, stride = %d\n", OC, stride);

    //printf("Any input file for input data?\n[filename]: ");
    std::string name = "../data/f32_2k2k16.in";
    //std::cin>>name;
    int fd = open(name.c_str(), O_RDONLY);
    void *in_ptr = nullptr;
    if(fd > 0)
        in_ptr = mmap(NULL, C*W*H*N*sizeof(Data_t), PROT_READ, MAP_SHARED,
                fd, 0);

    /* allocate memory */
#ifdef NHWC
    Halide::Runtime::Buffer<Data_t> input((Data_t*)in_ptr, C, W, H, N);
    Halide::Runtime::Buffer<Data_t> filter(nullptr, C, FW, FH, OC);
#else
    Halide::Runtime::Buffer<Data_t> input((Data_t*)in_ptr, W, H, C, N);
    Halide::Runtime::Buffer<Data_t> filter(nullptr, FW, FH, C, OC);
#endif

    const int OW = std::ceil((W - FW)/stride + 1);
    const int OH = std::ceil((H - FH)/stride + 1);
#ifdef NHWC
    Halide::Runtime::Buffer<Data_t> output(nullptr, OC, OW, OH, N);
#else
    Halide::Runtime::Buffer<Data_t> output(nullptr, OW, OH, OC, N);
#endif

    if(!in_ptr)
        input.allocate();
    filter.allocate();
    output.allocate();

    /* give initial value */
    printf("Generaing input...\n");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 65535.0);
    if(!in_ptr)
        input.for_each_value([&](Data_t &x)
                {
                x = static_cast<Data_t>(dis(gen));
                });
    filter.for_each_value([&](Data_t &x)
            {
            x = static_cast<Data_t>(dis(gen));
            });
    printf("Done!\n");

    /* running benchmark */
    printf("Running Benchmark!");
    double time = Halide::Tools::benchmark([&]()
            {
                int err = 0;
                for(int i=0;i<runs;i++)
                    err = Convolution(input, filter, stride, output);
                if(err)
                printf("\033[31mPipeline failed!\033[0m\n");
            });
    printf("...done, time: %g s\n", time);
    size_t total = 1;
    printf("\033[32mfp32 flops: \033[34m%.3f \033[32mmflops in NHWC\033[0m\n",
            (total * C * N * OH * OW * OC * FH * FW * 2) / 
                (time / runs * 1000 * 1000));

    /* check correctness */
    printf("Success!\n");
    return 0;
}

int main(int argc, char * argv[])
{
#ifdef USE_PPROF
    ProfilerStart("check.prof");
#endif
    profile(48, 128, 56, 88, 1, 1);
    profile(56, 128, 64, 80, 3, 3);
    profile(24, 3, 256, 320, 3, 2);
    profile(16, 3, 224, 352, 5, 1);
    profile(16, 3, 256, 320, 7, 2);
    profile(8, 8, 56, 88, 3, 1);
    profile(8, 8, 7, 11, 3, 1);
    profile(4, 4, 64, 80, 3, 1);
    profile(108, 108, 7, 7, 3, 1);
    profile(54, 54, 7, 7, 3, 1);
    profile(3, 3, 128, 128, 3, 1);
    profile(3, 3, 112, 112, 3, 1);
    profile(25, 16, 1920, 1080, 7, 2, 5);
#ifdef USE_PPROF
    ProfilerStop();
#endif
}
