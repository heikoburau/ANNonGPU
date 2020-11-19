// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// KullbackLeibler.cu.template
// ***********************************************************

#ifndef LEAN_AND_MEAN

#include "network_functions/KullbackLeibler.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"

#include <cstring>
#include <math.h>


namespace ann_on_gpu {

namespace kernel {


template<bool compute_gradient, typename Psi_t, typename PsiPrime_t, typename Ensemble>
void kernel::KullbackLeibler::compute_averages(
    Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble
) const {
    const auto this_ = *this;
    const auto psi_kernel = psi.kernel();
    const auto psi_prime_kernel = psi_prime.kernel();

    ensemble.foreach(
        psi_prime,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const typename Ensemble::Basis_t& configuration,
            const complex_t& log_psi_prime,
            typename PsiPrime_t::Payload& payload_prime,
            const double& weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t::dtype    log_psi;
            SHARED typename Psi_t::Payload  payload;
            SHARED double                   prob_ratio;

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.log_psi_s(log_psi, configuration, payload);

            SINGLE {
                prob_ratio = exp(2.0 * (log_psi.real() - log_psi_prime.real()));
                generic_atomicAdd(this_.log_ratio, weight * prob_ratio * (log_psi_prime - log_psi));
                generic_atomicAdd(this_.log_ratio_abs2, weight * prob_ratio * abs2(log_psi_prime - log_psi));
            }
            SYNC;

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    configuration,
                    payload_prime,
                    [&](const unsigned int k, const complex_t& O_k_element) {
                        generic_atomicAdd(
                            &this_.O_k[k],
                            weight * prob_ratio * conj(O_k_element)
                        );
                        generic_atomicAdd(
                            &this_.log_ratio_O_k[k],
                            weight * prob_ratio * (log_psi_prime - log_psi) * conj(O_k_element)
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
        log_ratio(1, gpu),
        log_ratio_abs2(1, gpu),
        O_k(num_params, gpu),
        log_ratio_O_k(num_params, gpu),
        prob_ratio(1, gpu)
    {
    this->gpu = gpu;

    this->kernel().log_ratio = this->log_ratio.data();
    this->kernel().log_ratio_abs2 = this->log_ratio_abs2.data();
    this->kernel().O_k = this->O_k.data();
    this->kernel().log_ratio_O_k = this->log_ratio_O_k.data();
    this->kernel().prob_ratio = this->prob_ratio.data();
}


void KullbackLeibler::clear() {
    this->log_ratio.clear();
    this->log_ratio_abs2.clear();
    this->O_k.clear();
    this->log_ratio_O_k.clear();
    this->prob_ratio.clear();
}


template<typename Psi_t, typename PsiPrime_t, typename Ensemble>
double KullbackLeibler::value(
    Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble
) {
    this->clear();
    this->compute_averages<false>(psi, psi_prime, ensemble);

    this->log_ratio.update_host();
    this->log_ratio_abs2.update_host();
    this->prob_ratio.update_host();

    cout << this->log_ratio.front() << endl;
    cout << this->log_ratio_abs2.front() << endl;
    cout << this->prob_ratio.front() << endl;

    this->log_ratio.front() /= this->prob_ratio.front();
    this->log_ratio_abs2.front() /= this->prob_ratio.front();

    return sqrt(max(
        1e-8,
        this->log_ratio_abs2.front() - abs2(this->log_ratio.front())
    ));

    // return this->log_ratio_abs2.front() - abs2(this->log_ratio.front());
}


template<typename Psi_t, typename PsiPrime_t, typename Ensemble>
double KullbackLeibler::gradient(
    complex<double>* result, Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble, const double nu
) {
    this->clear();

    // std::cout << psi.gpu << std::endl;
    // std::cout << psi_prime.gpu << std::endl;
    // std::cout << ensemble.gpu << std::endl;

    // std::cout << std::endl;

    // std::cout << psi.get_width() << std::endl;
    // std::cout << psi_prime.get_width() << std::endl;

    this->compute_averages<true>(psi, psi_prime, ensemble);

    // std::cout << std::endl;

    // std::cout << this->log_ratio.gpu << std::endl;
    // std::cout << this->log_ratio_abs2.gpu << std::endl;
    // std::cout << this->O_k.gpu << std::endl;
    // std::cout << this->log_ratio_O_k.gpu << std::endl;

    // std::cout << std::endl;

    // std::cout << this->log_ratio.size() << std::endl;
    // std::cout << this->log_ratio_abs2.size() << std::endl;
    // std::cout << this->O_k.size() << std::endl;
    // std::cout << this->log_ratio_O_k.size() << std::endl;

    // std::cout << std::endl;

    // std::cout << this->log_ratio.data() << std::endl;
    // std::cout << this->log_ratio_abs2.data() << std::endl;
    // std::cout << this->O_k.data() << std::endl;
    // std::cout << this->log_ratio_O_k.data() << std::endl;

    this->log_ratio.update_host();
    this->log_ratio_abs2.update_host();
    this->O_k.update_host();
    this->log_ratio_O_k.update_host();
    this->prob_ratio.update_host();

    this->log_ratio.front() /= this->prob_ratio.front();
    this->log_ratio_abs2.front() /= this->prob_ratio.front();
    for(auto k = 0u; k < this->num_params; k++) {
        this->O_k[k] /= this->prob_ratio.front();
        this->log_ratio_O_k[k] /= this->prob_ratio.front();
    }


    const auto value = sqrt(max(
        1e-8,
        this->log_ratio_abs2.front() - abs2(this->log_ratio.front())
    ));
    const auto factor = pow(value, nu);

    for(auto k = 0u; k < this->num_params; k++) {
        result[k] = (
            this->log_ratio_O_k[k] - this->log_ratio.front() * this->O_k[k]
        ).to_std() / factor;
    }

    return value;
}


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<Spins>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<PauliString>&);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double);
#endif


} // namespace ann_on_gpu

#endif // LEAN_AND_MEAN
