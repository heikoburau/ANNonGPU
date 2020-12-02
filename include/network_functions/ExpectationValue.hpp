#pragma once

#include "operators.hpp"
#include "Array.hpp"
#include "types.h"
#include <complex>

namespace ann_on_gpu {


class ExpectationValue {
private:
    Array<complex_t>    A_local;
    Array<double>       A_local_abs2;
    Array<double>       prob_ratio;

public:
    ExpectationValue(const bool gpu);

    template<typename Psi_t, typename Ensemble>
    complex<double> operator()(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename PsiSampling_t, typename Ensemble>
    complex<double> operator()(const Operator_t& operator_, Psi_t& psi, PsiSampling_t& psi_sampling, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    pair<double, complex<double>> fluctuation(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Ensemble>
    inline complex<double> __call__(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble) {
        return (*this)(operator_, psi, ensemble);
    }

    template<typename Psi_t, typename PsiSampling_t, typename Ensemble>
    inline complex<double> __call__(const Operator_t& operator_, Psi_t& psi, PsiSampling_t& psi_sampling, Ensemble& ensemble) {
        return (*this)(operator_, psi, psi_sampling, ensemble);
    }

#endif  // __PYTHONCC__

};

} // namespace ann_on_gpu
