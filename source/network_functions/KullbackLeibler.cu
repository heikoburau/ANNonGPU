#include "network_functions/KullbackLeibler.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"

#include <cstring>
#include <math.h>


namespace ann_on_gpu {

namespace kernel {


template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename Ensemble>
void kernel::KullbackLeibler::compute_averages(
    const Psi_t& psi, const Psi_t_prime& psi_prime, Ensemble& ensemble
) const {
    const auto this_ = *this;
    const auto psi_prime_kernel = psi_prime.kernel();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const typename Ensemble::Basis_t& configuration,
            const complex_t log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, configuration, activations);

            SINGLE
            {
                generic_atomicAdd(this_.log_ratio, weight * (log_psi_prime - log_psi));
                generic_atomicAdd(this_.log_ratio_abs2, weight * abs2(log_psi_prime - log_psi));
            }

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    configuration,
                    activations,
                    [&](const unsigned int k, const complex_t& O_k_element) {
                        generic_atomicAdd(
                            &this_.O_k[k],
                            weight * conj(O_k_element)
                        );
                        generic_atomicAdd(
                            &this_.log_ratio_O_k[k],
                            weight * (log_psi_prime - log_psi) * conj(O_k_element)
                        );
                    }
                );
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

} // namespace kernel

KullbackLeibler::KullbackLeibler(const unsigned int num_params, const bool gpu)
      : num_params(num_params),
        log_ratio_ar(1, gpu),
        log_ratio_abs2_ar(1, gpu),
        O_k_ar(num_params, gpu),
        log_ratio_O_k_ar(num_params, gpu)
    {
    this->gpu = gpu;

    this->log_ratio = this->log_ratio_ar.data();
    this->log_ratio_abs2 = this->log_ratio_abs2_ar.data();
    this->O_k = this->O_k_ar.data();
    this->log_ratio_O_k = this->log_ratio_O_k_ar.data();
}


void KullbackLeibler::clear() {
    this->log_ratio_ar.clear();
    this->log_ratio_abs2_ar.clear();
    this->O_k_ar.clear();
    this->log_ratio_O_k_ar.clear();
}


template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
double KullbackLeibler::value(
    const Psi_t& psi, const Psi_t_prime& psi_prime, Ensemble& ensemble
) {
    this->clear();
    this->compute_averages<false>(psi, psi_prime, ensemble);

    this->log_ratio_ar.update_host();
    this->log_ratio_abs2_ar.update_host();

    return sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));
}


template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
double KullbackLeibler::gradient(
    complex<double>* result, const Psi_t& psi, const Psi_t_prime& psi_prime, Ensemble& ensemble, const double nu
) {
    this->clear();
    this->compute_averages<true>(psi, psi_prime, ensemble);

    this->log_ratio_ar.update_host();
    this->log_ratio_abs2_ar.update_host();
    this->O_k_ar.update_host();
    this->log_ratio_O_k_ar.update_host();

    auto value = sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));
    value = max(value, 1e-6);
    const auto factor = pow(value, nu);

    for(auto k = 0u; k < this->num_params; k++) {
        result[k] = (
            this->log_ratio_O_k_ar[k] - this->log_ratio_ar.front() * this->O_k_ar[k]
        ).to_std() / factor;
    }

    return value;
}


// #ifdef ENABLE_MONTE_CARLO


// #ifdef ENABLE_SPINS

// template double KullbackLeibler::value(
//     const PsiClassical& psi, const PsiDeep& psi_prime, MonteCarloSpins& ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, MonteCarloSpins& ensemble, const double nu
// );

// #endif // ENABLE_SPINS

// #ifdef ENABLE_PAULIS

// template double KullbackLeibler::value(
//     const PsiClassical& psi, const PsiDeep& psi_prime, MonteCarloPaulis& ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, MonteCarloPaulis& ensemble, const double nu
// );

// #endif // ENABLE_PAULIS

// #endif // ENABLE_MONTE_CARLO


// #ifdef ENABLE_EXACT_SUMMATION

// #ifdef ENABLE_SPINS

// template double KullbackLeibler::value(
//     const PsiClassical& psi, const PsiDeep& psi_prime, ExactSummationSpins& ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, ExactSummationSpins& ensemble, const double nu
// );

// #endif // ENABLE_SPINS

// #ifdef ENABLE_PAULIS

// template double KullbackLeibler::value(
//     const PsiClassical& psi, const PsiDeep& psi_prime, ExactSummationPaulis& ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, ExactSummationPaulis& ensemble, const double nu
// );

// #endif // ENABLE_PAULIS

// #endif // ENABLE_EXACT_SUMMATION


} // namespace ann_on_gpu
