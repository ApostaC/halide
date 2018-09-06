#include "Halide.h"
using namespace Halide;
using Halide::Generator;
using Halide::Var;

class OpenCLTest : public Generator<OpenCLTest>
{
    public:
        Input<Buffer<int>> A{"A",1};
        Input<Buffer<int>> B{"B",1};
        Output<Buffer<int>> out{"C",1};
        GeneratorParam<int> size{"size", 1024};

        Func C;
        void generate()
        {
            Var x, xi, xo, xii;
            C(x) = A(x) * B(x);//host(x);
            out = C.in();

            out
                .split(x, xo, xi, 32 * 8)
                .split(xi, xi, xii, 32)
                .reorder(xi, xii, xo)
                .unroll(xi)
                .gpu_blocks(xo)
                .gpu_threads(xii);
            
            C.compute_at(out, xii)
                .unroll(x);

        }
};

HALIDE_REGISTER_GENERATOR(OpenCLTest, OpenCLTest)
