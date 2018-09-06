#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#ifdef X86
#include "bin/mat_mul.h"
#endif
#ifdef ARM
#include "bin/mat_mul_arm.h"
#endif
#include "halide_benchmark.h"
#include "HalideBuffer.h"

#ifdef CMP_2_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

using Halide::Runtime::Buffer;
using Halide::Tools::benchmark;

int main(int argc, char **argv) {
    int size = 1024;
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    printf("Before!\n");
    int a;
    scanf("%d", &a);
    // Check correctness using small-integer matrices
    if (1) {
        printf("generating input...\n");
        /*
        std::string name = "../data/f32_2k2k3.in";
        int fd = open(name.c_str(), O_RDONLY);
        float *ptr;
        if(fd<0) ptr = nullptr;
        else 
        {
            ptr = (float *)mmap(NULL, size*size, PROT_READ, MAP_SHARED, fd, 0);
            printf("input data file opened successfully!\n");
        }
        */
        Buffer<float> A(size, size), B(size, size), C(size, size);
        A.for_each_value([](float &v) {v = rand() & 3;});
        B.for_each_value([](float &v) {v = rand() & 3;});
        A.set_host_dirty();
        B.set_host_dirty();
        printf("in to mat mul kernel\n");
        mat_mul(A, B, C);
        printf("Kernel Finished!\n");
        C.copy_to_host();
        printf("Copy answer to host finished!\n");

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                float correct = 0.f;
                for (int k = 0; k < size; k++) {
                    correct += A(x, k) * B(k, y);
                }
                float actual = C(x, y);
                if (correct != actual) {
                    printf("%d %d: %f vs %f\n", x, y, correct, actual);
                    return -1;
                }
            }
        }
    }

    printf("Benching!\n");
    // Benchmark it
    {
        Buffer<float> A(size, size), B(size, size), C(size, size);
        double t = Halide::Tools::benchmark(3, 3, [&]() {
                mat_mul(A, B, C);
                C.device_sync();
                });
        double speed = size * size * size / (t * 1e9);
        printf("Halide time: %f, speed = %lf Gflops\n", t, speed);
    }

#ifdef CMP_2_CUBLAS
    // Benchmark cublas
    {
        float *A, *B, *C;
        cudaMalloc((void **)&A, size*size*4);
        cudaMalloc((void **)&B, size*size*4);
        cudaMalloc((void **)&C, size*size*4);
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 1.0f;
        double t = Halide::Tools::benchmark(3, 3, [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size, &alpha, A, size, B, size, &beta, C, size);
                cudaDeviceSynchronize();
                });
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cublasDestroy(handle);
        printf("cublas time: %f\n", t);
    }
#endif
    return 0;
}
