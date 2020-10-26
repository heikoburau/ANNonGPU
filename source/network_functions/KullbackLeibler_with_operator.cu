// #include "network_functions/KullbackLeibler.hpp"
// #include "ensembles.hpp"
// #include "quantum_states.hpp"

// #include <cstring>
// #include <math.h>


// namespace ann_on_gpu {

// namespace kernel {


// template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
// void kernel::KullbackLeibler::compute_averages(
//     const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& op,
//     const bool is_unitary, SpinEnsemble& spin_ensemble
// ) const {
//     const auto this_ = *this;
//     const auto psi_kernel = psi.kernel();
//     const auto psi_prime_kernel = psi_prime.kernel();

//     spin_ensemble.foreach(
//         psi,
//         [=] __device__ __host__ (
//             const unsigned int spin_index,
//             const Spins spins,
//             const complex_t log_psi,
//             typename Psi_t::dtype* angles,
//             typename Psi_t::dtype* activations,
//             const double weight
//         ) {
//             #include "cuda_kernel_defines.h"

//             SHARED complex_t log_psi_prime;
//             psi_prime_kernel.log_psi_s(log_psi_prime, spins, activations);

//             SHARED typename Psi_t::dtype local_energy;
//             op.local_energy(local_energy, psi_kernel, spins, log_psi, angles, activations);

//             SHARED complex_t diff;

//             SINGLE
//             {
//                 diff = log_psi_prime - log_psi - (is_unitary ? log(local_energy) : local_energy);
//                 generic_atomicAdd(this_.log_ratio, weight * diff);
//                 generic_atomicAdd(this_.log_ratio_abs2, weight * abs2(diff));
//             }

//             if(compute_gradient) {
//                 psi_prime_kernel.foreach_O_k(
//                     spins,
//                     activations,
//                     [&](const unsigned int k, const complex_t& O_k_element) {
//                         generic_atomicAdd(
//                             &this_.O_k[k],
//                             weight * conj(O_k_element)
//                         );
//                         generic_atomicAdd(
//                             &this_.log_ratio_O_k[k],
//                             weight * diff * conj(O_k_element)
//                         );
//                     }
//                 );
//             }
//         },
//         max(psi.get_width(), psi_prime.get_width())
//     );
// }

// } // namespace kernel


// template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
// double KullbackLeibler::value(
//     const Psi_t& psi, const Psi_t_prime& psi_prime,
//     const Operator& operator_, const bool is_unitary,
//     SpinEnsemble& spin_ensemble
// ) {
//     this->clear();
//     this->compute_averages<false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

//     this->log_ratio_ar.update_host();
//     this->log_ratio_abs2_ar.update_host();

//     this->log_ratio_ar.front() /= spin_ensemble.get_num_steps();
//     this->log_ratio_abs2_ar.front() /= spin_ensemble.get_num_steps();

//     return sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));
// }


// template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
// double KullbackLeibler::gradient(
//     complex<double>* result, const Psi_t& psi, const Psi_t_prime& psi_prime,
//     const Operator& operator_, const bool is_unitary,
//     SpinEnsemble& spin_ensemble, const double nu
// ) {
//     this->clear();
//     this->compute_averages<true>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

//     this->log_ratio_ar.update_host();
//     this->log_ratio_abs2_ar.update_host();
//     this->O_k_ar.update_host();
//     this->log_ratio_O_k_ar.update_host();

//     this->log_ratio_ar.front() /= spin_ensemble.get_num_steps();
//     this->log_ratio_abs2_ar.front() /= spin_ensemble.get_num_steps();

//     auto value = sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));
//     value = max(value, 1e-6);
//     const auto factor = pow(value, nu);

//     for(auto k = 0u; k < this->num_params; k++) {
//         this->O_k_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();
//         this->log_ratio_O_k_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();

//         result[k] = (
//             this->log_ratio_O_k_ar[k] - this->log_ratio_ar.front() * this->O_k_ar[k]
//         ).to_std() / factor;
//     }

//     return value;
// }



// #ifdef ENABLE_MONTE_CARLO

// #ifdef ENABLE_PSI_DEEP

// template double KullbackLeibler::value(
//     const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const bool is_unitary, MonteCarloLoop& spin_ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const bool is_unitary, MonteCarloLoop& spin_ensemble, const double nu
// );

// #endif

// #endif


// #ifdef ENABLE_EXACT_SUMMATION

// #ifdef ENABLE_PSI_DEEP

// template double KullbackLeibler::value(
//     const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const bool is_unitary, ExactSummation& spin_ensemble
// );
// template double KullbackLeibler::gradient(
//     complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const bool is_unitary, ExactSummation& spin_ensemble, const double nu
// );

// #endif

// #endif


// } // namespace ann_on_gpu
