#pragma once

#include "types.h"


namespace ann_on_gpu {


template<typename T>
HDINLINE
T my_logcosh(const T z, const unsigned int layer) {
    // using scalar = typename T::value_type;

    // const auto r = abs(z);
    // if(r < b) {
    //     return complex_t(0.0, 0.0);
    // }
    // return z * (1.0 - b / r);

    // const complex_t z2 = z * z;
    // return (1.0 / 180.0) * z2 * (90.0 - 15.0 * z2 + 4.0 * z2 * z2);

    // return sqrt(0.30102999566 + z*z);

    // return z*z * (1.0 / 2.0) - z*z*z*z * (1.0 / 12.0);
    // return z;

    // return sqrt(1.0 + z*z) - 1.0;

    // return complex_t(0.0, -1.0) * log(cosh(z)) + complex_t(0.0, -0.346574);

    // for TDVP it is important to use log cosh, or at least not Pade(2, 4)
    // if(layer == 0u) {
    //     return log(cosh(z));
    // }
    // else {
    //    return tanh(z);
    // }

    const auto z2 = z * z;
    const auto z4 = z2 * z2;

    if(layer == 0u) {
        // return 0.5 * z2 + (1.0 / 6.0) * z2 * z - (1.0 / 12.0) * z4 + (1.0 / 45.0) * z4 * z2;
        return 0.5 * z2 - (1.0 / 12.0) * z4 + (1.0 / 45.0) * z4 * z2;
        // return -(1.0 / 3.0) * z2 * z + (2.0 / 15.0) * z4 * z;
    }
    return z - (1.0 / 3.0) * z2 * z + (2.0 / 15.0) * z4 * z;

    // seems to be dangerous. Does not work for a SW-generator applied on an initial state.
    // return log(1.0 + z*z);

    // return 2.0 + 2.0 * z*z;
    // return log(1.0 + exp(z));

    // return 0.6932 + 0.5 * z * z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return ((1.0 / 2) * z - (1.0 / 120) * z2) / (1.0 - (1.0 / 10) * z + (1.0 / 120) * z2);
    // }

    // return log((exp(z) - 1.0) / z);

    // Pade(2, 4) is good for compression
    // const auto sign = z.real() > scalar(0.0) ? scalar(1.0) : scalar(-1.0);

    // return sign * z + (1.81168 - sign * 1.22741 * z) / (2.61371 + z * (sign * 2.0 + z)) - 0.693147;

    // return sign * scalar(0.9003320053750442) * z + (
    //     scalar(5.49914721954) - sign * scalar(2.16564366435) * z
    // ) / (
    //     scalar(9.19376335670885) + z * (sign * scalar(10.2180213465) + z * (scalar(7.771429504240965) + z * (sign * scalar(3.746646023906276) + z)))
    // ) - scalar(0.598139);
}

template<typename T>
HDINLINE
T my_tanh(const T z, const unsigned int layer) {
    // using scalar = typename T::value_type;
    // const auto r = abs(z);
    // if(r < b) {
    //     return complex_t(0.0, 0.0);
    // }
    // return complex_t(1.0, 0.0);// - complex_t(0.5 * b * r, 0.0) / z;

    // const complex_t z2 = z * z;
    // return (1.0 / 15.0) * z * (15.0 - 5.0 * z2 + 2.0 * z2 * z2);

    // return z / (0.30102999566 + z*z);
    // return z / (1.0 + z*z);
    // return complex_t(0.0, -1.0) * tanh(z);
    // return complex_t(2.0, 0.0) / (cosh(2.0 * z) + 1.0);
    // return 2.0 * z / (1.0 + z*z);

    // return z - z*z*z * (1.0 / 3.0);
    // return complex_t(1.0, 0.0);

    // return z / sqrt(1.0 + z*z);

    const auto z2 = z * z;
    const auto z4 = z2 * z2;

    // for TDVP it is important to use tanh, or at least not Pade(2, 4)
    if(layer == 0u) {
        // return tanh(z);
        // return z + 0.5 * z2 - (1.0 / 3.0) * z2 * z + (2.0 / 15.0) * z4 * z;
        return z - (1.0 / 3.0) * z2 * z + (2.0 / 15.0) * z4 * z;
        // return -z2 + (2.0 / 3.0) * z4;
    }
    else {
        // const auto co = cosh(z);
        // return 1.0 / (co * co);
        return 1.0 - z2 + (2.0 / 3.0) * z4;
    }

    // const auto e_z = exp(z);
    // return e_z + (1.0 + e_z);
    // return z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return (1.0 / 2 + (1.0 / 12) * z + (1.0 / 120) * z2) / (1.0 + (1.0 / 60) * z2);
    // }

    // const auto exp_z = exp(z);
    // return -1.0 / z + exp_z / (exp_z - 1.0);

    // Pade(2, 4) is good for compression
    // const auto sign = z.real() > scalar(0.0) ? scalar(1.0) : scalar(-1.0);
    // const auto denominator = 2.61371 + z * (sign * 2.0 + z);
    // return (
    //     z * (6.83146 + z * (sign * 10.4548 + z * (4.0 + sign * z)))
    // ) / (denominator * denominator);

    // const auto denominator = scalar(9.19376335670885) + z * (sign * scalar(10.218021346543315) + z * (scalar(7.771429504240965) + z * (sign * scalar(3.746646023906276) + z)));
    // return (
    //     z * (
    //         scalar(83.68563506532087) + z * (
    //             sign * scalar(177.6769746361748) + z * (
    //                 scalar(199.24474920889975) + z * (
    //                     sign * scalar(146.36284300074402) + z * (
    //                         scalar(70.82878897882324) + z * (
    //                             sign * scalar(26.632014683761202) + z * (
    //                                 scalar(6.746450656267947) + sign * scalar(0.9003320053750442) * z
    //     )))))))
    // ) / (denominator * denominator);
}

} // namespace ann_on_gpu
