#include "network_functions/HilbertSpaceDistance.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"

#include <cstring>
#include <math.h>


namespace ann_on_gpu {

namespace kernel {


template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename Ensemble>
void kernel::HilbertSpaceDistance::compute_averages(
    const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& operator_,
    const bool is_unitary, Ensemble& ensemble
) const {
    const auto this_ = *this;
    const auto psi_kernel = psi.kernel();
    const auto psi_prime_kernel = psi_prime.kernel();
    const auto N = psi.get_num_input_units();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype& log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"
            using dtype = typename Psi_t::dtype;
            using real_dtype = typename Psi_t::real_dtype;

            SHARED dtype local_energy;
            operator_.local_energy(local_energy, psi_kernel, configuration, log_psi, angles, activations);

            SHARED dtype log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, configuration, activations);
            SHARED dtype       omega;
            SHARED real_dtype  probability_ratio;

            SINGLE
            {
                if(is_unitary) {
                    omega = exp(conj(log_psi_prime - log_psi)) * local_energy;
                    generic_atomicAdd(
                        this_.next_state_norm_avg,
                        weight * (local_energy * conj(local_energy)).real()
                    );
                }
                else {
                    omega = exp(local_energy + conj(log_psi_prime - log_psi));
                    generic_atomicAdd(
                        this_.next_state_norm_avg,
                        weight * exp(real_dtype(2.0) * local_energy.real())
                    );
                }
                probability_ratio = exp(real_dtype(2.0) * (log_psi_prime.real() - log_psi.real()));

                generic_atomicAdd(this_.omega_avg, weight * precision_cast<complex_t>(omega));
                generic_atomicAdd(this_.probability_ratio_avg, weight * probability_ratio);
            }

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    configuration,
                    activations,
                    [&](const unsigned int k, const dtype& O_k_element) {
                        generic_atomicAdd(&this_.omega_O_k_avg[k], weight * precision_cast<complex_t>(omega * conj(O_k_element)));
                        generic_atomicAdd(&this_.probability_ratio_O_k_avg[k], weight * precision_cast<complex_t>(probability_ratio * conj(O_k_element)));
                    }
                );
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

} // namespace kernel

HilbertSpaceDistance::HilbertSpaceDistance(const unsigned int num_params, const bool gpu)
    :
    num_params(num_params),
    omega_avg_ar(1, gpu),
    omega_O_k_avg_ar(num_params, gpu),
    probability_ratio_avg_ar(1, gpu),
    probability_ratio_O_k_avg_ar(num_params, gpu),
    next_state_norm_avg_ar(1, gpu)
{
    this->gpu = gpu;

    this->omega_avg = this->omega_avg_ar.data();
    this->omega_O_k_avg = this->omega_O_k_avg_ar.data();
    this->probability_ratio_avg = this->probability_ratio_avg_ar.data();
    this->probability_ratio_O_k_avg = this->probability_ratio_O_k_avg_ar.data();
    this->next_state_norm_avg = this->next_state_norm_avg_ar.data();
}


void HilbertSpaceDistance::clear() {
    this->omega_avg_ar.clear();
    this->omega_O_k_avg_ar.clear();
    this->probability_ratio_avg_ar.clear();
    this->probability_ratio_O_k_avg_ar.clear();
    this->next_state_norm_avg_ar.clear();
}


template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
double HilbertSpaceDistance::distance(
    const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& operator_, const bool is_unitary,
    Ensemble& ensemble
) {
    this->clear();
    this->compute_averages<false>(psi, psi_prime, operator_, is_unitary, ensemble);

    this->omega_avg_ar.update_host();
    this->probability_ratio_avg_ar.update_host();
    this->next_state_norm_avg_ar.update_host();

    const auto u = abs2(this->omega_avg_ar.front());
    const auto v = this->next_state_norm_avg_ar.front() * this->probability_ratio_avg_ar.front();

    return sqrt(max(1.0 - u / v, 1e-8));
}


template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
double HilbertSpaceDistance::gradient(
    complex<double>* result, const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& operator_,
    const bool is_unitary, Ensemble& ensemble, const float nu
) {
    this->clear();
    this->compute_averages<true>(psi, psi_prime, operator_, is_unitary, ensemble);

    this->omega_avg_ar.update_host();
    this->omega_O_k_avg_ar.update_host();
    this->probability_ratio_avg_ar.update_host();
    this->probability_ratio_O_k_avg_ar.update_host();
    this->next_state_norm_avg_ar.update_host();

    const auto u = (this->omega_avg_ar.front() * conj(this->omega_avg_ar.front())).real();
    const auto v = this->next_state_norm_avg_ar.front() * this->probability_ratio_avg_ar.front();
    const auto distance = sqrt(max(1.0 - u / v, 1e-8));
    const auto prefactor = pow(distance, nu);

    for(auto k = 0u; k < this->num_params; k++) {
        const auto u_k_prime = conj(this->omega_avg_ar.front()) * this->omega_O_k_avg_ar[k];
        const auto v_k_prime = this->next_state_norm_avg_ar.front() * this->probability_ratio_O_k_avg_ar[k];

        result[k] = (
            -(u_k_prime * v - u * v_k_prime) / (v * v)
        ).to_std() / prefactor;
    }

    return distance;
}


#ifdef ENABLE_MONTE_CARLO

#ifdef ENABLE_SPINS
template double HilbertSpaceDistance::distance(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, MonteCarloSpins& ensemble);
template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, MonteCarloSpins& ensemble, const float nu);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template double HilbertSpaceDistance::distance(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, MonteCarloPaulis& ensemble);
template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, MonteCarloPaulis& ensemble, const float nu);
#endif // ENABLE_PAULIS

#endif // ENABLE_MONTE_CARLO


#ifdef ENABLE_EXACT_SUMMATION

#ifdef ENABLE_SPINS
template double HilbertSpaceDistance::distance(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, ExactSummationSpins& ensemble);
template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, ExactSummationSpins& ensemble, const float nu);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template double HilbertSpaceDistance::distance(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, ExactSummationPaulis& ensemble);
template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary, ExactSummationPaulis& ensemble, const float nu);
#endif // ENABLE_PAULIS

#endif // ENABLE_EXACT_SUMMATION


} // namespace ann_on_gpu
