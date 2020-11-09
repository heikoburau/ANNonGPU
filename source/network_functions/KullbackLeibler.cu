// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// KullbackLeibler.cu.template
// ***********************************************************

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
            typename Psi_t::Payload& payload,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t_prime::dtype    log_psi_prime;
            SHARED typename Psi_t_prime::Payload  payload_prime;

            psi_prime_kernel.init_payload(payload_prime, configuration);
            psi_prime_kernel.log_psi_s(log_psi_prime, configuration, payload_prime);

            SINGLE
            {
                generic_atomicAdd(this_.log_ratio, weight * (log_psi_prime - log_psi));
                generic_atomicAdd(this_.log_ratio_abs2, weight * abs2(log_psi_prime - log_psi));
            }

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    configuration,
                    payload_prime,
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


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<1u>&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<2u>&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<1u>&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<2u>&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(const PsiDeep&, const PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, const PsiDeep&, const PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&, const double);
#endif


} // namespace ann_on_gpu
