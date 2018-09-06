#include "Halide.h"


using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;
using Halide::ConciseCasts::f32;
using Halide::ConciseCasts::f64;
//using Halide::ConciseCasts::u8_sat;
//using Halide::ConciseCasts::i16;
using namespace Halide;

typedef float Data_t;
typedef double DoubleWided_t;
#define UPPER_CAST f64
#define LOWER_CAST f32 
class Convolution_NHWC : public Generator<Convolution_NHWC>
{
    public:
        /* Format is CWHN, which equals to 
         * NHWC arrangement in memory 
         */
        Input<Buffer<Data_t>> input_{"input",4};

        Input<Buffer<Data_t>> filter_{"filter",4};

        Input<int> stride_ { "stride" };

        Output<Buffer<Data_t>> output_{"output",4};

        GeneratorParam<bool> usingGPU{"usingGPU", false};

        void generate()
        {
            /* x->width y->height */
            Var x("x"), y("y"), batch("batch"), channel("channel");
            Func bounded_input = 
                constant_exterior(input_, LOWER_CAST(0),
                        { { Expr(), Expr() },
                          { 0, input_.dim(1).extent() },
                          { 0, input_.dim(2).extent() },
                          { Expr(), Expr() } });
            Func convolved("convolved");
            RDom filter_dom(0, filter_.dim(0).extent(),         //ic
                            0, filter_.dim(1).extent(),         //fw
                            0, filter_.dim(2).extent());        //fh

            Func tempConv("tempConv");

            Var ic, fw, fh;
            tempConv(ic, fw, fh, channel, x, y, batch) =
                UPPER_CAST(filter_(ic, fw, fh, channel)) *
                UPPER_CAST(bounded_input(ic, x*stride_ + fw, y*stride_ + fh, batch));

            convolved(channel, x, y, batch) += 
                tempConv(filter_dom[0], filter_dom[1], filter_dom[2],
                        channel, x, y, batch);
                /*
                UPPER_CAST(filter_(filter_dom[0], filter_dom[1],
                                      filter_dom[2], channel)) *
                UPPER_CAST(bounded_input(filter_dom[0],
                            x*stride_ + filter_dom[1],
                            y*stride_ + filter_dom[2],
                            batch));
                            */

            output_(channel, x, y, batch) = LOWER_CAST(convolved(channel, x, y, batch));

            /* schedule */
            {
                Var c_out, c_in, xi, xo, fuse_c;
                int vector_size_u8 = get_target().natural_vector_size<Data_t>();
                Expr can_vec = filter_.dim(3).extent() >= vector_size_u8;
                printf("vector size is %d\n", vector_size_u8);
                Expr unroll_3 = filter_.dim(3).extent() % 3 == 0,
                     unroll_2 = filter_.dim(3).extent() % 2 == 0;


                output_.parallel(y)
                    .split(x, xo, xi, vector_size_u8, Halide::TailStrategy::GuardWithIf)
                    .vectorize(xi);

                output_.specialize(unroll_3)
                    .split(channel, c_out, c_in, 3)
                    .unroll(c_in);
                output_.specialize(unroll_2)
                    .split(channel, c_out, c_in, 2)
                    .unroll(c_in);
                output_.specialize(unroll_2 && unroll_3)
                    .split(channel, c_out, c_in, 6)
                    .unroll(c_in);
            }
            /* TODO: modify here? */
            //bounded_input.compute_at(output_, batch);
        }



};


class Convolution_NCHW : public Halide::Generator<Convolution_NCHW>
{
    public:
        /* Format is WHCN, which equals to 
         * NHWC arrangement in memory 
         */
        Input<Buffer<Data_t>> input_{"input",4};

        Input<Buffer<Data_t>> filter_{"filter",4};

        Input<int> stride_ { "stride" };

        Output<Buffer<Data_t>> output_{"output",4};

        GeneratorParam<bool> usingGPU{"usingGPU", false};

        void generate()
        {
            /* x->width y->height */
            Var x("x"), y("y"), batch("batch"), channel("channel");
            Func bounded_input = 
                constant_exterior(input_, LOWER_CAST(0),
                        { { 0, input_.dim(0).extent() },
                          { 0, input_.dim(1).extent() },
                          { Expr(), Expr() },
                          { Expr(), Expr() } });

            /*
            RDom filter_dom(0, filter_.dim(0).extent(), 
                            0, filter_.dim(1).extent());
            RDom step_dom(0,filter_.dim(2).extent());

            Var step;
            Func tempConv("TempConv");
            tempConv(x, y, channel, batch, step)
                += UPPER_CAST(filter_(filter_dom[0], filter_dom[1],
                                        step, channel)) *
                  UPPER_CAST(bounded_input(
                              x*stride_ + filter_dom[0],
                              y*stride_ + filter_dom[1],
                              step, batch));
                                           
            Func convolved("convolved");
            convolved(x, y, channel, batch) += tempConv(x, y, channel, batch, step_dom);

            output_(x, y, channel, batch) = LOWER_CAST(convolved(x, y, channel, batch));
            */

            Func convolved("convolved");
            RDom filter_dom(0, filter_.dim(0).extent(), 
                            0, filter_.dim(1).extent(),
                            0, filter_.dim(2).extent());

            Func tempConv("tempConv");

            Var ic, fw, fh;
            tempConv(fw, fh, ic, x, y, channel, batch) =
                UPPER_CAST(filter_(fw, fh, ic, channel)) *
                UPPER_CAST(bounded_input(x*stride_ + fw, y*stride_ + fh, ic, batch));

            convolved(x, y, channel, batch) +=
                tempConv(filter_dom[0], filter_dom[1], filter_dom[2],
                        x, y, channel, batch);
            /*
                UPPER_CAST(filter_(filter_dom[0], filter_dom[1],
                                      filter_dom[2], channel)) *
                UPPER_CAST(bounded_input(
                            x*stride_ + filter_dom[0],
                            y*stride_ + filter_dom[1],
                            filter_dom[2],
                            batch));
                            */
            output_(x, y, channel, batch) = LOWER_CAST(convolved(x, y, channel, batch));

            /* schedule */
            {
                Var c_out, c_in;
                Var xo, yo, xi, xii, yi, yii, tile_index;
                int vector_size_u8 = get_target().natural_vector_size<Data_t>();
                const int BIG_TILE = 64,
                      SMALL_TILE_X = vector_size_u8,
                      SMALL_TILE_Y = 2;
                printf("vector size is %d\n", vector_size_u8);

                output_.reorder(channel, x, y, batch)
                    .parallel(y)
                    .vectorize(x, vector_size_u8, Halide::TailStrategy::GuardWithIf)
                    .unroll(channel, 3, Halide::TailStrategy::GuardWithIf)
                    ;
            }
            /* TODO: modify here? */
            //bounded_input.compute_at(output_, batch);
        }

};

HALIDE_REGISTER_GENERATOR(Convolution_NHWC, Convolution_NHWC)
HALIDE_REGISTER_GENERATOR(Convolution_NCHW, Convolution_NCHW)
