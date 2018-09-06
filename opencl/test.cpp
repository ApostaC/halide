#include <stdio.h>
#include <stdlib.h>
#include "HalideRuntimeOpenCL.h"
#include "HalideBuffer.h"

#ifdef ARM
#include "bin/test_arm.h"
#endif

#ifdef X86
#include "bin/test_x86.h"
#endif

#ifndef SIZE
    #define SIZE 256
#endif

int a[SIZE], b[SIZE] ;
int init()
{
    for(int i=0;i<SIZE;i++)
    {
        a[i] = i;
        b[i] = SIZE-i;
    }
    return 0;
}

void debugPrint(const char *name, int * data, int size)
{
    printf("data in %s\n",name);
    for(int i=0;i<size;i++)
        printf("%d ",data[i]);
    printf("\n");
}

int main()
{
    init();
    printf("Got size: %d\n", SIZE);
    Halide::Runtime::Buffer<int> inputA(a, SIZE), inputB(b, SIZE), 
        output(SIZE);
    output.allocate();
    Halide::Runtime::Buffer<int> outHost(*output.raw_buffer(),
            Halide::Runtime::BufferDeviceOwnership::AllocatedDeviceAndHost);
    Test(inputA, inputB, outHost);
    output.copy_from(outHost);
    for(int i=0;i<SIZE;i++)
        if(output(i) != i * (SIZE-i))
            printf("Not equal at index:%d , expect: %d, but gives: %d\n",
                   i, i*(SIZE-i), output(i));
    printf("Finished!\n");
    return 0;
}
