#pragma once

#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <complex>
#include <tuple>
// #include <memory>


namespace ann_on_gpu {

namespace kernel {

struct KullbackLeibler {
    bool gpu;

    complex_t*  deviation;
    double*     deviation2;
    complex_t*  O_k;
    complex_t*  deviation_O_k_conj;

    // for estimating the noise
    double*     deviation2_O_k2;
    complex_t*  deviation_O_k;
    complex_t*  deviation2_O_k;
    complex_t*  deviation_O_k2;
    double*     O_k2;

    complex_t*  mean_deviation;
    complex_t*  last_mean_deviation;

    double*     prob_ratio;


    template<bool compute_gradient, bool noise, typename Psi_t, typename Psi_t_prime, typename Ensemble>
    void compute_averages(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, double threshold
    ) const;

    inline KullbackLeibler& kernel() {
        return *this;
    }
};

} // namespace kernel


struct KullbackLeibler : public kernel::KullbackLeibler {
    const unsigned int  num_params;

    Array<complex_t>  deviation;
    Array<double>     deviation2;
    Array<complex_t>  O_k;
    Array<complex_t>  deviation_O_k_conj;

    Array<double>     deviation2_O_k2;
    Array<complex_t>  deviation_O_k;
    Array<complex_t>  deviation2_O_k;
    Array<complex_t>  deviation_O_k2;
    Array<double>     O_k2;

    Array<complex_t>  mean_deviation;
    Array<complex_t>  last_mean_deviation;

    Array<double>     prob_ratio;

    void clear();

    KullbackLeibler(const unsigned int num_params, const bool gpu);

    void update_last_mean_deviation();

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    double value(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, double threshold
    );

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    double gradient(
        complex<double>* result, Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, const double nu, double threshold
    );

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    tuple<
        Array<complex_t>,
        Array<double>,
        double
    > gradient_with_noise(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& ensemble, const double nu, double threshold
    );

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
    pair<xt::pytensor<complex<double>, 1u>, double> gradient_py(
        Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& spin_ensemble, const double nu, double threshold
    ) {
        xt::pytensor<complex<double>, 1u> grad(std::array<long int, 1u>({(long int)psi_prime.num_params}));

        const double value = this->gradient(grad.data(), psi, psi_prime, spin_ensemble, nu, threshold);

        return {grad, value};
    }

#endif // __PYTHONCC__

};

} // namespace ann_on_gpu
