#include "Halide.h"

using namespace Halide;

using Data_t = float;

class SimpleMMGen : public Halide::Generator<SimpleMMGen>
{
    public:

        Input<Buffer<Data_t>> A{"A",2};
        Input<Buffer<Data_t>> B{"B",2};
        GeneratorParam<int> size{"size", 1024};
        Output<Buffer<Data_t>> out{"out",2};

        Func prod;
        Var x,y;

        void generate()
        {
            const int wrap_size = 16;
            RDom r(0, size);
            prod(x, y) += A(x, r) * B(r, y);
            out = prod.in();
            Var xi, yi, xio, xii, yii, xo;

            out.bound(x, 0, size)
                .bound(y, 0, size)
                .tile(x, y, xi, yi, 8*wrap_size, 8)
                .split(xi, xio, xii, wrap_size)
                .reorder(xio, yi, xii, x, y)
                .unroll(xio)
                .unroll(yi)
                .gpu_blocks(x, y).gpu_threads(xii);
            prod
                .compute_at(out, xii)
                .unroll(x)
                .unroll(y)
                .update()
                .unroll(r, 2)
                .reorder(y, x, r)
                .unroll(x)
                .unroll(y);
                ;
            B.in()
                .compute_at(prod, y)
                .vectorize(B.in().args()[0])
                ;
        }

};

HALIDE_REGISTER_GENERATOR(SimpleMMGen, SimpleMMGen)
