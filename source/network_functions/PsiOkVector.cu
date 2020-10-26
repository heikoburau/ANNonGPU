// #include "network_functions/PsiOkVector.hpp"
// #include "quantum_states.hpp"
// #include "ensembles.hpp"
// #include "types.h"

// namespace ann_on_gpu {


// template<typename Psi_t>
// void psi_O_k_vector(complex<double>* result, const Psi_t& psi, const Spins& spins) {
//     complex_t* result_ptr;
//     auto O_k_length = psi.num_params;
//     auto psi_kernel = psi.kernel();

//     // printf("O_k_length: %d\n", O_k_length);
//     MALLOC(result_ptr, sizeof(complex_t) * O_k_length, psi.gpu);
//     // MEMSET(result_ptr, 0, sizeof(complex_t) * O_k_length, psi.gpu);

//     const auto functor = [=] __host__ __device__ () {
//         #include "cuda_kernel_defines.h"

//         SHARED typename Psi_t::dtype activations[Psi_t::max_width];

//         psi_kernel.foreach_O_k(
//             spins,
//             activations,
//             [&](const unsigned int k, const typename Psi_t::dtype& O_k_element) {
//                 // printf("%d, %f, %f\n", k, O_k_element.real(), O_k_element.imag());
//                 result_ptr[k] = precision_cast<complex_t>(O_k_element);
//             }
//         );
//     };

//     if(psi.gpu) {
//         cuda_kernel<<<1, psi.get_width()>>>(functor);
//     }
//     else {
//         functor();
//     }

//     MEMCPY_TO_HOST(result, result_ptr, sizeof(complex_t) * O_k_length, psi.gpu);
//     FREE(result_ptr, psi.gpu);
// }


// template<typename Psi_t, typename SpinEnsemble>
// void psi_O_k_vector(complex<double>* result, complex<double>* result_std, const Psi_t& psi, SpinEnsemble& spin_ensemble) {
//     const auto O_k_length = psi.num_params;
//     const auto psi_kernel = psi.kernel();

//     complex_t* result_device;
//     complex_t* result2_device;

//     MALLOC(result_device, sizeof(complex_t) * O_k_length, psi.gpu);
//     MALLOC(result2_device, sizeof(complex_t) * O_k_length, psi.gpu);
//     MEMSET(result_device, 0, sizeof(complex_t) * O_k_length, psi.gpu);
//     MEMSET(result2_device, 0, sizeof(complex_t) * O_k_length, psi.gpu);

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
//             psi_kernel.foreach_O_k(
//                 spins,
//                 activations,
//                 [&](const unsigned int k, const complex_t& O_k_element) {
//                     generic_atomicAdd(&result_device[k], weight * O_k_element);
//                     const auto O_k_element2 = complex_t(
//                         O_k_element.real() * O_k_element.real(), O_k_element.imag() * O_k_element.imag()
//                     );
//                     generic_atomicAdd(&result2_device[k], weight * O_k_element2);
//                 }
//             );
//         }
//     );

//     MEMCPY_TO_HOST(result, result_device, sizeof(complex_t) * O_k_length, psi.gpu);
//     MEMCPY_TO_HOST(result_std, result2_device, sizeof(complex_t) * O_k_length, psi.gpu);
//     FREE(result_device, psi.gpu);
//     FREE(result2_device, psi.gpu);

//     for(auto k = 0u; k < O_k_length; k++) {
//         result[k] /= spin_ensemble.get_num_steps();
//         result_std[k] /= spin_ensemble.get_num_steps();

//         result_std[k] = result_std[k] - complex<double>(
//             result[k].real() * result[k].real(), result[k].imag() * result[k].imag()
//         );
//     }
// }


// template<typename Psi_t, typename SpinEnsemble>
// pair<Array<complex_t>, Array<double>> psi_O_k_vector(const Psi_t& psi, SpinEnsemble& spin_ensemble) {
//     const auto O_k_length = psi.num_params;
//     const auto psi_kernel = psi.kernel();

//     Array<complex_t> result(O_k_length, psi.gpu);
//     Array<double> result_std(O_k_length, psi.gpu);

//     result.clear();
//     result_std.clear();

//     auto result_ptr = result.data();
//     auto result_std_ptr = result_std.data();

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
//             psi_kernel.foreach_O_k(
//                 spins,
//                 activations,
//                 [&](const unsigned int k, const complex_t& O_k_element) {
//                     generic_atomicAdd(&result_ptr[k], weight * O_k_element);
//                     generic_atomicAdd(&result_std_ptr[k], weight * (O_k_element * conj(O_k_element)).real());
//                 }
//             );
//         }
//     );

//     result.update_host();
//     result_std.update_host();

//     for(auto k = 0u; k < O_k_length; k++) {
//         result[k] /= spin_ensemble.get_num_steps();
//         result_std[k] /= spin_ensemble.get_num_steps();

//         result_std[k] = sqrt((result_std[k] - result[k] * conj(result[k])).real());
//     }

//     return {result, result_std};
// }

// #ifdef ENABLE_PSI_DEEP
// template void psi_O_k_vector(complex<double>* result, const PsiDeep& psi, const Spins& spins);
// #endif // ENABLE_PSI_DEEP

// #ifdef ENABLE_PSI_PAIR
// // template void psi_O_k_vector(complex<double>* result, const PsiPair& psi, const Spins& spins);
// #endif // ENABLE_PSI_PAIR


// } // namespace ann_on_gpu
