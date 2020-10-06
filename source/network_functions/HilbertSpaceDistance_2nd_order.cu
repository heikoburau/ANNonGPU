// #include "network_functions/HilbertSpaceDistance.hpp"
// #include "ensembles.hpp"
// #include "quantum_states.hpp"

// #include <cstring>
// #include <math.h>


// namespace ann_on_gpu {

// namespace kernel {


// template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
// void kernel::HilbertSpaceDistance::compute_averages_2nd_order(
//     const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& op, const Operator& op2, SpinEnsemble& spin_ensemble
// ) const {
//     const auto num_params = psi_prime.num_params;

//     const auto this_ = *this;
//     const auto psi_kernel = psi.kernel();
//     const auto psi_prime_kernel = psi_prime.kernel();
//     const auto N = psi.get_num_input_units();

//     spin_ensemble.foreach(
//         psi,
//         [=] __device__ __host__ (
//             const unsigned int spin_index,
//             const Spins spins,
//             const typename Psi_t::dtype& log_psi,
//             typename Psi_t::Angles& angles,
//             const double weight
//         ) {
//             #include "cuda_kernel_defines.h"
//             using dtype = typename Psi_t::dtype;
//             using real_dtype = typename Psi_t::real_dtype;

//             SHARED dtype local_energy;
//             op.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

//             SHARED dtype local_energy2;
//             op2.local_energy(local_energy2, psi_kernel, spins, log_psi, angles);

//             SHARED typename Psi_t_prime::Angles angles_prime;
//             angles_prime.init(psi_prime_kernel, spins);

//             SHARED dtype log_psi_prime;
//             psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

//             SHARED dtype        omega;
//             SHARED real_dtype   probability_ratio;

//             SINGLE
//             {
//             omega = exp(local_energy + typename Psi_t::real_dtype(0.5f) * (local_energy2 - local_energy * local_energy) + conj(log_psi_prime - log_psi));
//                 generic_atomicAdd(
//                     this_.next_state_norm_avg,
//                     weight * exp(real_dtype(2.0) * (
//                         local_energy + real_dtype(0.5) * (local_energy2 - local_energy * local_energy)
//                     ).real())
//                 );

//                 probability_ratio = exp(real_dtype(2.0) * (log_psi_prime.real() - log_psi.real()));

//                 generic_atomicAdd(this_.omega_avg, weight * precision_cast<complex_t>(omega));
//                 generic_atomicAdd(this_.probability_ratio_avg, weight * probability_ratio);
//             }
//         },
//         max(psi.get_width(), psi_prime.get_width())
//     );
// }

// } // namespace kernel


// template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
// double HilbertSpaceDistance::distance_2nd_order(
//     const Psi_t& psi, const Psi_t_prime& psi_prime, const Operator& op, const Operator& op2,
//     SpinEnsemble& spin_ensemble
// ) {
//     this->clear();
//     this->compute_averages_2nd_order(psi, psi_prime, op, op2, spin_ensemble);

//     this->omega_avg_ar.update_host();
//     this->probability_ratio_avg_ar.update_host();
//     this->next_state_norm_avg_ar.update_host();

//     this->omega_avg_ar.front() /= spin_ensemble.get_num_steps();
//     this->probability_ratio_avg_ar.front() /= spin_ensemble.get_num_steps();
//     this->next_state_norm_avg_ar.front() /= spin_ensemble.get_num_steps();

//     return sqrt(
//         1.0 - (this->omega_avg_ar.front() * conj(this->omega_avg_ar.front())).real() / (
//             this->next_state_norm_avg_ar.front() *this->probability_ratio_avg_ar.front()
//         )
//     );
// }



// #ifdef ENABLE_MONTE_CARLO

// // #ifdef ENABLE_PSI

// // template double HilbertSpaceDistance::distance_2nd_order(const Psi& psi, const Psi& psi_prime, const Operator& op, const Operator& op2, MonteCarloLoop& spin_ensemble);

// // #endif

// #ifdef ENABLE_PSI_DEEP

// template double HilbertSpaceDistance::distance_2nd_order(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, MonteCarloLoop& spin_ensemble);

// #endif

// // #ifdef ENABLE_PSI_PAIR

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiPair& psi, const PsiPair& psi_prime, const Operator& op, const Operator& op2, MonteCarloLoop& spin_ensemble);

// // #endif

// // #if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_DEEP)

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiClassical& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, MonteCarloLoop& spin_ensemble);

// // #endif

// // #if defined(ENABLE_PSI_EXACT) && defined(ENABLE_PSI_DEEP)

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiExact& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, MonteCarloLoop& spin_ensemble);

// // #endif


// #endif



// #ifdef ENABLE_EXACT_SUMMATION

// // #ifdef ENABLE_PSI

// // template double HilbertSpaceDistance::distance_2nd_order(const Psi& psi, const Psi& psi_prime, const Operator& op, const Operator& op2, ExactSummation& spin_ensemble);

// // #endif

// #ifdef ENABLE_PSI_DEEP

// template double HilbertSpaceDistance::distance_2nd_order(const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, ExactSummation& spin_ensemble);

// #endif

// // #ifdef ENABLE_PSI_PAIR

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiPair& psi, const PsiPair& psi_prime, const Operator& op, const Operator& op2, ExactSummation& spin_ensemble);

// // #endif

// // #if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_DEEP)

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiClassical& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, ExactSummation& spin_ensemble);

// // #endif

// // #if defined(ENABLE_PSI_EXACT) && defined(ENABLE_PSI_DEEP)

// // template double HilbertSpaceDistance::distance_2nd_order(const PsiExact& psi, const PsiDeep& psi_prime, const Operator& op, const Operator& op2, ExactSummation& spin_ensemble);

// // #endif

// #endif


// } // namespace ann_on_gpu
