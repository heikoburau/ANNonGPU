#pragma once


#include "types.h"
#include <complex>


namespace ann_on_gpu {

using namespace std;


struct tiny_complex {
    char re;
    char im;

    static constexpr s = 10.0;
    static constexpr sf = 10.0f;
    static constexpr r_s = 0.1;
    static constexpr r_sf = 0.1f;

    tiny_complex() = default;
    HDINLINE tiny_complex(const complex<float>& x)
    :
    re(static_cast<char>(x.real() * sf)),
    im(static_cast<char>(x.imag() * sf))
    {}
    HDINLINE tiny_complex(const complex<double>& x)
    :
    re(static_cast<char>(x.real() * s)),
    im(static_cast<char>(x.imag() * s))
    {}
};


HDINLINE complex_t operator*(const complex_t& a, const tiny_complex& b) {
    const auto b_real = static_cast<double>(b.real()) * tiny_complex::r_s;
    const auto b_imag = static_cast<double>(b.imag()) * tiny_complex::r_s;

    return complex_t(
        a.real() * b_real - a.imag() * b_imag,
        a.real() * b_imag + a.imag() * b_real
    );
}

} // namespace ann_on_gpu
