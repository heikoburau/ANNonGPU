#pragma once

#include "operator/Operator.hpp"
#include "types.h"
#include <complex>

namespace ann_on_gpu {


class ExpectationValue {
private:
    Array<complex_t>    A_local;
    Array<double>       A_local_abs2;

public:
    ExpectationValue(const bool gpu);

    template<typename Psi_t, typename Ensemble>
    complex<double> operator()(const Psi_t& psi, const Operator& operator_, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    pair<double, complex<double>> fluctuation(const Psi_t& psi, const Operator& operator_, Ensemble& ensemble);

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Ensemble>
    inline complex<double> __call__(const Psi_t& psi, const Operator& operator_, Ensemble& ensemble) {
        return (*this)(psi, operator_, ensemble);
    }

#endif  // __PYTHONCC__

};

} // namespace ann_on_gpu
