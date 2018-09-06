#include "Halide.h"
#include <iostream>

using namespace Halide;

int res_buffer[1000];
void conv_1d(int input_size, int FILTER_SIZE)
{
    Var x("x");
    Func input, filter, output;
    input(x) = x;
    filter(x) = FILTER_SIZE-x;

    input.compute_root();
    filter.compute_root();

    /*============DEFINITION===============*/
    Func bounded_input;
    bounded_input = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_size}});
    RDom fdom(0,FILTER_SIZE);
    output(x) += bounded_input(x+fdom) * filter(fdom);


    /*============SCHEDULING==============*/
    Var fdi, fdo;
    output.update(0)
        .allow_race_conditions()
        .parallel(fdom);

    /*============CALCULATOR==============*/
    Buffer<int> out_buf(1000);
    output.realize(out_buf);

    for(int i=0;i<1000;i++)
    {
        if(out_buf(i) != res_buffer[i])
            fprintf(stderr,"Not correct at %d, res_buffer:%d -- output:%d\n",
                    i,res_buffer[i],out_buf(i));
    }
    output.compile_to_lowered_stmt("test.stmt",{},Text);
}

int main()
{
    int buf[10];
    for(int i=0;i<10;i++) buf[i] = 10-i;
    for(int i=0;i<1000;i++)
    {
        res_buffer[i] = 0 ;
        for(int r=0;r<10 && i+r < 1000;r++)
            res_buffer[i] += (i+r)*buf[r];
    }
    conv_1d(1000, 10);
}
