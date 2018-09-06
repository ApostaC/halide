#include "Halide.h"

using namespace Halide;

const int size = 16;
int main()
{
    Var x("x"), y("y"), xi("xi"), xo("xo"), yi("yi"), yo("yo");
    Func prod, out;
    prod(x, y) = x + y;
    out(x, y) = prod(x,y);

    out.tile(x, y, xo, yo, xi, yi, 4, 4)
        .vectorize(xi)
        .unroll(yi);
    prod.compute_at(out, xo)
        .unroll(x)
        .unroll(y);

    out.compile_to_lowered_stmt("prod-cons.stmt", {}, Text);
    auto output = out.realize(32, 32);
}
