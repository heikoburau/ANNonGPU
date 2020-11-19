#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <complex>
// #include <memory>


namespace ann_on_gpu {

namespace kernel {

class KullbackLeibler {
public:
    bool gpu;

    complex_t*  log_ratio;
    double*     log_ratio_abs2;
    complex_t*  O_k;
    complex_t*  log_ratio_O_k;
    double*     prob_ratio;

    template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename Ensemble>
    void compute_averages(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble
    ) const;

    inline KullbackLeibler& kernel() {
        return *this;
    }
};

} // namespace kernel


class KullbackLeibler : public kernel::KullbackLeibler {
private:
    const unsigned int  num_params;

    Array<complex_t> log_ratio;
    Array<double>    log_ratio_abs2;
    Array<complex_t> O_k;
    Array<complex_t> log_ratio_O_k;
    Array<double>    prob_ratio;

    void clear();

public:
    KullbackLeibler(const unsigned int num_params, const bool gpu);

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    double value(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble
    );

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    double gradient(
        complex<double>* result, Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, const double nu
    );

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    pair<xt::pytensor<complex<double>, 1u>, double> gradient_py(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, const double nu
    ) {
        xt::pytensor<complex<double>, 1u> grad(std::array<long int, 1u>({(long int)psi_prime.num_params}));

        const double value = this->gradient(grad.data(), psi, psi_prime, spin_ensemble, nu);

        return {grad, value};
    }

#endif // __PYTHONCC__

};

} // namespace ann_on_gpu
