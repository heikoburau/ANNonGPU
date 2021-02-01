// An implementation of C++ std::complex for use on CUDA devices.
// Written by John C. Travers <jtravs@gmail.com> (2012)
//
// Missing:
//  - long double support (not supported on CUDA)
//  - some integral pow functions (due to lack of C++11 support on CUDA)
//
// Heavily derived from the LLVM libcpp project (svn revision 147853).
// Based on libcxx/include/complex.
// The git history contains the complete change history from the original.
// The modifications are licensed as per the original LLVM license below.
//
// -*- C++ -*-
//===--------------------------- complex ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <complex>

#ifndef CUDA_COMPLEX_HPP
#define CUDA_COMPLEX_HPP

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__ __forceinline__
#else
#define CUDA_CALLABLE_MEMBER inline
#endif

/*
    complex synopsis

template<class T>
class complex
{
public:
    typedef T value_type;

    complex(const T& re = T(), const T& im = T());
    complex(const complex&);
    template<class X> complex(const complex<X>&);

    T real() const;
    T imag() const;

    void real(T);
    void imag(T);

    complex<T>& operator= (const T&);
    complex<T>& operator+=(const T&);
    complex<T>& operator-=(const T&);
    complex<T>& operator*=(const T&);
    complex<T>& operator/=(const T&);

    complex& operator=(const complex&);
    template<class X> complex<T>& operator= (const complex<X>&);
    template<class X> complex<T>& operator+=(const complex<X>&);
    template<class X> complex<T>& operator-=(const complex<X>&);
    template<class X> complex<T>& operator*=(const complex<X>&);
    template<class X> complex<T>& operator/=(const complex<X>&);
};

template<>
class complex<float>
{
public:
    typedef float value_type;

    constexpr complex(float re = 0.0f, float im = 0.0f);
    explicit constexpr complex(const complex<double>&);

    constexpr float real() const;
    void real(float);
    constexpr float imag() const;
    void imag(float);

    complex<float>& operator= (float);
    complex<float>& operator+=(float);
    complex<float>& operator-=(float);
    complex<float>& operator*=(float);
    complex<float>& operator/=(float);

    complex<float>& operator=(const complex<float>&);
    template<class X> complex<float>& operator= (const complex<X>&);
    template<class X> complex<float>& operator+=(const complex<X>&);
    template<class X> complex<float>& operator-=(const complex<X>&);
    template<class X> complex<float>& operator*=(const complex<X>&);
    template<class X> complex<float>& operator/=(const complex<X>&);
};

template<>
class complex<double>
{
public:
    typedef double value_type;

    constexpr complex(double re = 0.0, double im = 0.0);
    constexpr complex(const complex<float>&);

    constexpr double real() const;
    void real(double);
    constexpr double imag() const;
    void imag(double);

    complex<double>& operator= (double);
    complex<double>& operator+=(double);
    complex<double>& operator-=(double);
    complex<double>& operator*=(double);
    complex<double>& operator/=(double);
    complex<double>& operator=(const complex<double>&);

    template<class X> complex<double>& operator= (const complex<X>&);
    template<class X> complex<double>& operator+=(const complex<X>&);
    template<class X> complex<double>& operator-=(const complex<X>&);
    template<class X> complex<double>& operator*=(const complex<X>&);
    template<class X> complex<double>& operator/=(const complex<X>&);
};

// 26.3.6 operators:
template<class T> complex<T> operator+(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&, const T&);
template<class T> complex<T> operator+(const T&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const T&);
template<class T> complex<T> operator-(const T&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const T&);
template<class T> complex<T> operator*(const T&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const T&);
template<class T> complex<T> operator/(const T&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&);
template<class T> bool operator==(const complex<T>&, const complex<T>&);
template<class T> bool operator==(const complex<T>&, const T&);
template<class T> bool operator==(const T&, const complex<T>&);
template<class T> bool operator!=(const complex<T>&, const complex<T>&);
template<class T> bool operator!=(const complex<T>&, const T&);
template<class T> bool operator!=(const T&, const complex<T>&);

template<class T, class charT, class traits>
  basic_istream<charT, traits>&
  operator>>(basic_istream<charT, traits>&, complex<T>&);
template<class T, class charT, class traits>
  basic_ostream<charT, traits>&
  operator<<(basic_ostream<charT, traits>&, const complex<T>&);

// 26.3.7 values:

template<class T>              T real(const complex<T>&);
                          double real(double);
template<Integral T>      double real(T);
                          float  real(float);

template<class T>              T imag(const complex<T>&);
                          double imag(double);
template<Integral T>      double imag(T);
                          float  imag(float);

template<class T> T abs(const complex<T>&);

template<class T>              T arg(const complex<T>&);
                          double arg(double);
template<Integral T>      double arg(T);
                          float  arg(float);

template<class T>              T norm(const complex<T>&);
                          double norm(double);
template<Integral T>      double norm(T);
                          float  norm(float);

template<class T>      complex<T>           conj(const complex<T>&);
                       complex<double>      conj(double);
template<Integral T>   complex<double>      conj(T);
                       complex<float>       conj(float);

template<class T>    complex<T>           proj(const complex<T>&);
                     complex<double>      proj(double);
template<Integral T> complex<double>      proj(T);
                     complex<float>       proj(float);

template<class T> complex<T> polar(const T&, const T& = 0);

// 26.3.8 transcendentals:
template<class T> complex<T> acos(const complex<T>&);
template<class T> complex<T> asin(const complex<T>&);
template<class T> complex<T> atan(const complex<T>&);
template<class T> complex<T> acosh(const complex<T>&);
template<class T> complex<T> asinh(const complex<T>&);
template<class T> complex<T> atanh(const complex<T>&);
template<class T> complex<T> cos (const complex<T>&);
template<class T> complex<T> cosh (const complex<T>&);
template<class T> complex<T> exp (const complex<T>&);
template<class T> complex<T> log (const complex<T>&);
template<class T> complex<T> log10(const complex<T>&);

template<class T> complex<T> pow(const complex<T>&, const T&);
template<class T> complex<T> pow(const complex<T>&, const complex<T>&);
template<class T> complex<T> pow(const T&, const complex<T>&);

template<class T> complex<T> sin (const complex<T>&);
template<class T> complex<T> sinh (const complex<T>&);
template<class T> complex<T> sqrt (const complex<T>&);
template<class T> complex<T> tan (const complex<T>&);
template<class T> complex<T> tanh (const complex<T>&);

template<class T, class charT, class traits>
  basic_istream<charT, traits>&
  operator>>(basic_istream<charT, traits>& is, complex<T>& x);

template<class T, class charT, class traits>
  basic_ostream<charT, traits>&
  operator<<(basic_ostream<charT, traits>& o, const complex<T>& x);

*/

#include <math.h>
#include <sstream>

namespace cuda_complex {

template<class _Tp> class  complex;

template<class _Tp> CUDA_CALLABLE_MEMBER complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
template<class _Tp> CUDA_CALLABLE_MEMBER complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template<class _Tp>
class  complex
{
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:

    complex() = default;

    template<class _Xp> CUDA_CALLABLE_MEMBER
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    CUDA_CALLABLE_MEMBER value_type real() const {return __re_;}
    CUDA_CALLABLE_MEMBER value_type imag() const {return __im_;}

    CUDA_CALLABLE_MEMBER void real(value_type __re) {__re_ = __re;}
    CUDA_CALLABLE_MEMBER void imag(value_type __im) {__im_ = __im;}

    CUDA_CALLABLE_MEMBER complex& operator= (const value_type& __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    CUDA_CALLABLE_MEMBER complex& operator+=(const value_type& __re) {__re_ += __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator-=(const value_type& __re) {__re_ -= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator*=(const value_type& __re) {__re_ *= __re; __im_ *= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator/=(const value_type& __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * __c;
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / __c;
            return *this;
        }
};

template<> class  complex<double>;

template<>
class  complex<float>
{
public:
    float __re_;
    float __im_;
public:
    typedef float value_type;

    complex() = default;
    /*constexpr*/ CUDA_CALLABLE_MEMBER complex(float __re, float __im)
        : __re_(__re), __im_(__im) {}
    CUDA_CALLABLE_MEMBER complex(double __re, double __im)
        : __re_(__re), __im_(__im) {}
    CUDA_CALLABLE_MEMBER complex(float x)
        : __re_(x), __im_(0.0f) {}
    CUDA_CALLABLE_MEMBER
    explicit /*constexpr*/ complex(const complex<double>& __c);

    inline complex(const std::complex<float>& z) : __re_(z.real()), __im_(z.imag()) {}

    /*constexpr*/ CUDA_CALLABLE_MEMBER float real() const {return __re_;}
    /*constexpr*/ CUDA_CALLABLE_MEMBER float imag() const {return __im_;}

    CUDA_CALLABLE_MEMBER void real(value_type __re) {__re_ = __re;}
    CUDA_CALLABLE_MEMBER void imag(value_type __im) {__im_ = __im;}

    CUDA_CALLABLE_MEMBER complex& operator= (float __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    CUDA_CALLABLE_MEMBER complex& operator+=(float __re) {__re_ += __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator-=(float __re) {__re_ -= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator*=(float __re) {__re_ *= __re; __im_ *= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator/=(float __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * __c;
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / __c;
            return *this;
        }

    template<class _Xp>
    CUDA_CALLABLE_MEMBER
    friend void atomicAdd(complex<_Xp>* address, const complex<_Xp>& value);

    std::complex<float> to_std() const {return std::complex<float>(this->__re_, this->__im_);}
};

template<>
class  complex<double>
{
public:
    double __re_;
    double __im_;
public:
    typedef double value_type;

    complex() = default;
    /*constexpr*/ CUDA_CALLABLE_MEMBER complex(double __re, double __im)
        : __re_(__re), __im_(__im) {}
    CUDA_CALLABLE_MEMBER complex(float __re, float __im)
        : __re_(__re), __im_(__im) {}
    CUDA_CALLABLE_MEMBER complex(double x)
        : __re_(x), __im_(0.0) {}

    inline complex(const std::complex<double>& z) : __re_(z.real()), __im_(z.imag()) {}

    /*constexpr*/ CUDA_CALLABLE_MEMBER double real() const {return __re_;}
    /*constexpr*/ CUDA_CALLABLE_MEMBER double imag() const {return __im_;}

    CUDA_CALLABLE_MEMBER void real(value_type __re) {__re_ = __re;}
    CUDA_CALLABLE_MEMBER void imag(value_type __im) {__im_ = __im;}

    CUDA_CALLABLE_MEMBER complex& operator= (double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    CUDA_CALLABLE_MEMBER complex& operator+=(double __re) {__re_ += __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator-=(double __re) {__re_ -= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator*=(double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    CUDA_CALLABLE_MEMBER complex& operator/=(double __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * __c;
            return *this;
        }
    template<class _Xp> CUDA_CALLABLE_MEMBER complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / __c;
            return *this;
        }

    template<class _Xp>
    CUDA_CALLABLE_MEMBER
    friend void atomicAdd(complex<_Xp>* address, const complex<_Xp>& value);

    std::complex<double> to_std() const {return std::complex<double>(this->__re_, this->__im_);}
};

//constexpr
// CUDA_CALLABLE_MEMBER
// complex<float>::complex(const complex<double>& __c)
//     : __re_(__c.real()), __im_(__c.imag()) {}

// //constexpr
// CUDA_CALLABLE_MEMBER
// complex<double>::complex(const complex<float>& __c)
//     : __re_(__c.real()), __im_(__c.imag()) {}

// 26.3.6 operators:

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t += __x;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(-__y);
    __t += __x;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    // return complex<_Tp>(
    //     __z.real() * __w.real() - __z.imag() * __w.imag(),
    //     __z.real() * __w.imag() + __z.imag() * __w.real()
    // );

    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    // if (isnan(__x) && isnan(__y))
    // {
    //     bool __recalc = false;
    //     if (isinf(__a) || isinf(__b))
    //     {
    //         __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
    //         __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
    //         if (isnan(__c))
    //             __c = copysign(_Tp(0), __c);
    //         if (isnan(__d))
    //             __d = copysign(_Tp(0), __d);
    //         __recalc = true;
    //     }
    //     if (isinf(__c) || isinf(__d))
    //     {
    //         __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
    //         __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
    //         if (isnan(__a))
    //             __a = copysign(_Tp(0), __a);
    //         if (isnan(__b))
    //             __b = copysign(_Tp(0), __b);
    //         __recalc = true;
    //     }
    //     if (!__recalc && (isinf(__ac) || isinf(__bd) ||
    //                       isinf(__ad) || isinf(__bc)))
    //     {
    //         if (isnan(__a))
    //             __a = copysign(_Tp(0), __a);
    //         if (isnan(__b))
    //             __b = copysign(_Tp(0), __b);
    //         if (isnan(__c))
    //             __c = copysign(_Tp(0), __c);
    //         if (isnan(__d))
    //             __d = copysign(_Tp(0), __d);
    //         __recalc = true;
    //     }
    //     if (__recalc)
    //     {
    //         __x = _Tp(INFINITY) * (__a * __c - __b * __d);
    //         __y = _Tp(INFINITY) * (__a * __d + __b * __c);
    //     }
    // }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator*(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t *= __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator*(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t *= __x;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();

    _Tp alpha = _Tp(1.0) / (__c * __c + __d * __d);

    return complex<_Tp>(
        (__a * __c + __b * __d) * alpha,
        (__b * __c - __a * __d) * alpha
    );
}

// template<class _Tp>
// CUDA_CALLABLE_MEMBER
// complex<_Tp>
// operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
// {
//     int __ilogbw = 0;

//     _Tp __a = __z.real();
//     _Tp __b = __z.imag();
//     _Tp __c = __w.real();
//     _Tp __d = __w.imag();

//     _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
//     if (isfinite(__logbw))
//     {
//         __ilogbw = static_cast<int>(__logbw);
//         __c = scalbn(__c, -__ilogbw);
//         __d = scalbn(__d, -__ilogbw);
//     }
//     _Tp alpha = _Tp(1.0) / (__c * __c + __d * __d);

//     return complex<_Tp>(
//         scalbn((__a * __c + __b * __d) * alpha, -__ilogbw),
//         scalbn((__b * __c - __a * __d) * alpha, -__ilogbw)
//     );
// }

// template<class _Tp>
// CUDA_CALLABLE_MEMBER
// complex<_Tp>
// operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
// {
//     int __ilogbw = 0;
//     _Tp __a = __z.real();
//     _Tp __b = __z.imag();
//     _Tp __c = __w.real();
//     _Tp __d = __w.imag();
//     _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
//     if (isfinite(__logbw))
//     {
//         __ilogbw = static_cast<int>(__logbw);
//         __c = scalbn(__c, -__ilogbw);
//         __d = scalbn(__d, -__ilogbw);
//     }
//     _Tp __denom = __c * __c + __d * __d;
//     _Tp __x = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
//     _Tp __y = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
//     if (isnan(__x) && isnan(__y))
//     {
//         if ((__denom == _Tp(0)) && (!isnan(__a) || !isnan(__b)))
//         {
//             __x = copysign(_Tp(INFINITY), __c) * __a;
//             __y = copysign(_Tp(INFINITY), __c) * __b;
//         }
//         else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
//         {
//             __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
//             __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
//             __x = _Tp(INFINITY) * (__a * __c + __b * __d);
//             __y = _Tp(INFINITY) * (__b * __c - __a * __d);
//         }
//         else if (isinf(__logbw) && __logbw > _Tp(0) && isfinite(__a) && isfinite(__b))
//         {
//             __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
//             __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
//             __x = _Tp(0) * (__a * __c + __b * __d);
//             __y = _Tp(0) * (__b * __c - __a * __d);
//         }
//     }
//     return complex<_Tp>(__x, __y);
// }

// template<>
// CUDA_CALLABLE_MEMBER
// complex<float>
// operator/(const complex<float>& __z, const complex<float>& __w)
// {
//     int __ilogbw = 0;
//     float __a = __z.real();
//     float __b = __z.imag();
//     float __c = __w.real();
//     float __d = __w.imag();
//     float __logbw = logbf(fmaxf(fabsf(__c), fabsf(__d)));
//     if (isfinite(__logbw))
//     {
//         __ilogbw = static_cast<int>(__logbw);
//         __c = scalbnf(__c, -__ilogbw);
//         __d = scalbnf(__d, -__ilogbw);
//     }
//     float __denom = __c * __c + __d * __d;
//     float __x = scalbnf((__a * __c + __b * __d) / __denom, -__ilogbw);
//     float __y = scalbnf((__b * __c - __a * __d) / __denom, -__ilogbw);
//     if (isnan(__x) && isnan(__y))
//     {
//         if ((__denom == float(0)) && (!isnan(__a) || !isnan(__b)))
//         {
// #ifdef __CUDACC__
// #pragma warning(suppress: 4756) // Ignore INFINITY related warning
// #endif // __CUDACC__
//             __x = copysignf(INFINITY, __c) * __a;
// #ifdef __CUDACC__
// #pragma warning(suppress: 4756) // Ignore INFINITY related warning
// #endif // __CUDACC__
//             __y = copysignf(INFINITY, __c) * __b;
//         }
//         else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
//         {
//             __a = copysignf(isinf(__a) ? float(1) : float(0), __a);
//             __b = copysignf(isinf(__b) ? float(1) : float(0), __b);
// #ifdef __CUDACC__
// #pragma warning(suppress: 4756) // Ignore INFINITY related warning
// #endif // __CUDACC__
//             __x = INFINITY * (__a * __c + __b * __d);
// #ifdef __CUDACC__
// #pragma warning(suppress: 4756) // Ignore INFINITY related warning
// #endif // __CUDACC__
//             __y = INFINITY * (__b * __c - __a * __d);
//         }
//         else if (isinf(__logbw) && __logbw > float(0) && isfinite(__a) && isfinite(__b))
//         {
//             __c = copysignf(isinf(__c) ? float(1) : float(0), __c);
//             __d = copysignf(isinf(__d) ? float(1) : float(0), __d);
//             __x = float(0) * (__a * __c + __b * __d);
//             __y = float(0) * (__b * __c - __a * __d);
//         }
//     }
//     return complex<float>(__x, __y);

// }

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator/(const complex<_Tp>& __x, const _Tp& __y)
{
    return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator/(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t /= __y;
    return __t;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator+(const complex<_Tp>& __x)
{
    return __x;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
operator-(const complex<_Tp>& __x)
{
    return complex<_Tp>(-__x.real(), -__x.imag());
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    return __x.real() == __y && __x.imag() == 0;
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    return __x == __y.real() && 0 == __y.imag();
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

// 26.3.7 values:

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
abs2(const complex<_Tp>& __c)
{
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

// real

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
real(const complex<_Tp>& __c)
{
    return __c.real();
}

CUDA_CALLABLE_MEMBER
double
real(double __re)
{
    return __re;
}

CUDA_CALLABLE_MEMBER
float
real(float  __re)
{
    return __re;
}

// imag

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
imag(const complex<_Tp>& __c)
{
    return __c.imag();
}

CUDA_CALLABLE_MEMBER
double
imag(double __re)
{
    return 0;
}

CUDA_CALLABLE_MEMBER
float
imag(float  __re)
{
    return 0;
}

// abs

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
abs(const complex<_Tp>& __c)
{
    return hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
arg(const complex<_Tp>& __c)
{
    return atan2(__c.imag(), __c.real());
}

CUDA_CALLABLE_MEMBER
double
arg(double __re)
{
    return atan2(0., __re);
}

CUDA_CALLABLE_MEMBER
float
arg(float __re)
{
    return atan2f(0.F, __re);
}

// norm

template<class _Tp>
CUDA_CALLABLE_MEMBER
_Tp
norm(const complex<_Tp>& __c)
{
    if (isinf(__c.real()))
        return fabs(__c.real());
    if (isinf(__c.imag()))
        return fabs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

CUDA_CALLABLE_MEMBER
double
norm(double __re)
{
    return __re * __re;
}

CUDA_CALLABLE_MEMBER
float
norm(float __re)
{
    return __re * __re;
}

// conj

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
conj(const complex<_Tp>& __c)
{
    return complex<_Tp>(__c.real(), -__c.imag());
}

CUDA_CALLABLE_MEMBER
double
conj(const double& __c)
{
    return __c;
}

// proj

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
proj(const complex<_Tp>& __c)
{
    complex<_Tp> __r = __c;
    if (isinf(__c.real()) || isinf(__c.imag()))
        __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
    return __r;
}

// polar

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
polar(const _Tp& __rho, const _Tp& __theta = _Tp(0))
{
    if (isnan(__rho) || signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (isnan(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (isinf(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * cos(__theta);
    if (isnan(__x))
        __x = 0;
    _Tp __y = __rho * sin(__theta);
    if (isnan(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
log(const complex<_Tp>& __x)
{
    return complex<_Tp>(::log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
log10(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(10));
}

// sqrt

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
sqrt(const complex<_Tp>& __x)
{
    if (isinf(__x.imag()))
        return complex<_Tp>(_Tp(INFINITY), __x.imag());
    if (isinf(__x.real()))
    {
        if (__x.real() > _Tp(0))
            return complex<_Tp>(__x.real(), isnan(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
        return complex<_Tp>(isnan(__x.imag()) ? __x.imag() : _Tp(0), copysign(__x.real(), __x.imag()));
    }
    return polar(::sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
exp(const complex<_Tp>& __x)
{
    _Tp __i = __x.imag();
    /*if (isinf(__x.real()))
    {
        if (__x.real() < _Tp(0))
        {
            if (!isfinite(__i))
                __i = _Tp(1);
        }
        else if (__i == 0 || !isfinite(__i))
        {
            if (isinf(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    else if (isnan(__x.real()) && __x.imag() == 0)
        return __x;*/
    _Tp __e = ::exp(__x.real());
    return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return exp(__y * log(__x));
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
pow(const complex<_Tp>& __x, const _Tp& __y)
{
    return pow(__x, complex<_Tp>(__y));
}

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
pow(const _Tp& __x, const complex<_Tp>& __y)
{
    return pow(complex<_Tp>(__x), __y);
}

// asinh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
asinh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return __x;
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__x.imag() == 0)
            return __x;
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(copysign(__x.imag(), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) + _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
acosh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(fabs(__x.real()), __x.imag());
        if (isinf(__x.imag())) {
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
            else
                return complex<_Tp>(-__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
        }
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(fabs(__x.imag()), __x.real());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(fabs(__x.imag()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) - _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
}

// atanh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
atanh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.imag()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (isnan(__x.imag()))
    {
        if (isinf(__x.real()) || __x.real() == 0)
            return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }
    if (isnan(__x.real()))
    {
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.real()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (fabs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
    {
        return complex<_Tp>(copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
    }
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
sinh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return __x;
    return complex<_Tp>(sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
cosh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(fabs(__x.real()), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return complex<_Tp>(fabs(__x.real()), __x.imag());
    return complex<_Tp>(::cosh(__x.real()) * ::cos(__x.imag()), ::sinh(__x.real()) * ::sin(__x.imag()));
}

// tanh

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
tanh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()))
    {
        if (!isfinite(__x.imag()))
            return complex<_Tp>(_Tp(1), _Tp(0));
        return complex<_Tp>(_Tp(1), copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
    }
    if (isnan(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __2r(_Tp(2) * __x.real());
    _Tp __2i(_Tp(2) * __x.imag());
    _Tp __d(::cosh(__2r) + ::cos(__2i));
    return  complex<_Tp>(::sinh(__2r)/__d, ::sin(__2i)/__d);
}

// asin

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
asin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
acos(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (isinf(__x.imag()))
        {
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
        return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    if (__x.real() == 0)
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    complex<_Tp> __z = log(__x + sqrt(pow(__x, _Tp(2)) - _Tp(1)));
    if (signbit(__x.imag()))
        return complex<_Tp>(fabs(__z.imag()), fabs(__z.real()));
    return complex<_Tp>(fabs(__z.imag()), -fabs(__z.real()));
}

// atan

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
atan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
sin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
cos(const complex<_Tp>& __x)
{
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template<class _Tp>
CUDA_CALLABLE_MEMBER
complex<_Tp>
tan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// atomicAdd

template<class _Tp>
CUDA_CALLABLE_MEMBER
void
atomicAdd(complex<_Tp>* address, const complex<_Tp>& value)
{
#ifdef __CUDA_ARCH__
    ::atomicAdd(&(address->__re_), value.real());
    ::atomicAdd(&(address->__im_), value.imag());
#else
    *address += value;
#endif // __CUDACC__
}

template<class _Tp, class _CharT, class _Traits>
std::basic_istream<_CharT, _Traits>&
operator>>(std::basic_istream<_CharT, _Traits>& __is, complex<_Tp>& __x)
{
    if (__is.good())
    {
        ws(__is);
        if (__is.peek() == _CharT('('))
        {
            __is.get();
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
            {
                ws(__is);
                _CharT __c = __is.peek();
                if (__c == _CharT(','))
                {
                    __is.get();
                    _Tp __i;
                    __is >> __i;
                    if (!__is.fail())
                    {
                        ws(__is);
                        __c = __is.peek();
                        if (__c == _CharT(')'))
                        {
                            __is.get();
                            __x = complex<_Tp>(__r, __i);
                        }
                        else
                            __is.setstate(std::ios_base::failbit);
                    }
                    else
                        __is.setstate(std::ios_base::failbit);
                }
                else if (__c == _CharT(')'))
                {
                    __is.get();
                    __x = complex<_Tp>(__r, _Tp(0));
                }
                else
                    __is.setstate(std::ios_base::failbit);
            }
            else
                __is.setstate(std::ios_base::failbit);
        }
        else
        {
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
                __x = complex<_Tp>(__r, _Tp(0));
            else
                __is.setstate(std::ios_base::failbit);
        }
    }
    else
        __is.setstate(std::ios_base::failbit);
    return __is;
}

template<class _Tp, class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
{
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.real() << ',' << __x.imag() << ')';
    return __os << __s.str();
}


template<typename T>
struct get_real_type {
    typedef typename T::value_type type;
};

template<>
struct get_real_type<double> {
    typedef double type;
};

template<>
struct get_real_type<float> {
    typedef float type;
};


inline std::complex<double> to_std(const complex<double>& c) {
    return std::complex<double>(c.real(), c.imag());
}

inline double to_std(const double& c) {
    return c;
}


} // close namespace cuda_complex

#endif  // CUDA_COMPLEX_HPP
