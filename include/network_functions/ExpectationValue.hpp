#pragma once

#include "operators.hpp"
#include "Array.hpp"
#include "types.h"
#include <complex>
#include <vector>

namespace ann_on_gpu {

using namespace std;


class ExpectationValue {
private:
    Array<complex_t>    A_local;
    Array<double>       A_local_abs2;
    Array<double>       prob_ratio;

public:
    ExpectationValue(const bool gpu);

    template<typename Psi_t, typename Ensemble>
    complex<double> operator()(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    complex<double> exp_sigma_z(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    Array<complex_t> operator()(const vector<Operator_t>& operator_array, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename PsiSampling_t, typename Ensemble>
    complex<double> operator()(const Operator_t& operator_, Psi_t& psi, PsiSampling_t& psi_sampling, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    pair<double, complex<double>> fluctuation(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    pair<Array<complex_t>, complex<double>> gradient(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    pair<Array<complex_t>, Array<double>> gradient_with_noise(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble);

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Ensemble>
    inline complex<double> __call__(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble) {
        return (*this)(operator_, psi, ensemble);
    }

    template<typename Psi_t, typename Ensemble>
    inline xt::pytensor<complex<double>, 1u> __call__array(const vector<Operator_t>& operator_, Psi_t& psi, Ensemble& ensemble) {
        return (*this)(operator_, psi, ensemble).to_pytensor_1d();
    }

    template<typename Psi_t, typename PsiSampling_t, typename Ensemble>
    inline complex<double> __call__(const Operator_t& operator_, Psi_t& psi, PsiSampling_t& psi_sampling, Ensemble& ensemble) {
        return (*this)(operator_, psi, psi_sampling, ensemble);
    }

    template<typename Psi_t, typename Ensemble>
    pair<xt::pytensor<complex<double>, 1u>, complex<double>> gradient_py(const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble) {
        auto gradient_and_value = this->gradient(operator_, psi, ensemble);

        return make_pair(gradient_and_value.first.to_pytensor_1d(), gradient_and_value.second);
    }

#endif  // __PYTHONCC__

};

} // namespace ann_on_gpu
