#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiVector.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t>
void psi_vector(complex<double>* result, const Psi_t& psi) {
    ExactSummation exact_summation(psi.get_num_input_units(), psi.gpu);

    complex_t* result_ptr;
    MALLOC(result_ptr, sizeof(complex_t) * exact_summation.get_num_steps(), psi.gpu);

    const auto log_prefactor = log(psi.prefactor);
    auto psi_kernel = psi.kernel();

    exact_summation.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const double weight
        ) {
            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                result_ptr[spin_index] = exp(log_prefactor + log_psi);
            }
        }
    );

    MEMCPY_TO_HOST(result, result_ptr, sizeof(complex_t) * exact_summation.get_num_steps(), psi.gpu);
    FREE(result_ptr, psi.gpu);
}

template<typename Psi_t>
Array<complex_t> psi_vector(const Psi_t& psi) {
    Array<complex_t> result(1 << psi.get_num_input_units(), false);
    psi_vector(reinterpret_cast<complex<double>*>(result.data()), psi);

    return result;
}


#ifdef ENABLE_PSI_DEEP
template void psi_vector(complex<double>* result, const PsiDeep& psi);
template Array<complex_t> psi_vector(const PsiDeep& psi);
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_DEEP_MIN
template void psi_vector(complex<double>* result, const PsiDeepMin& psi);
template Array<complex_t> psi_vector(const PsiDeepMin& psi);
#endif // ENABLE_PSI_DEEP_MIN

#ifdef ENABLE_PSI_PAIR
template void psi_vector(complex<double>* result, const PsiPair& psi);
template Array<complex_t> psi_vector(const PsiPair& psi);
#endif // ENABLE_PSI_PAIR

#ifdef ENABLE_PSI_CLASSICAL
template void psi_vector(complex<double>* result, const PsiClassical& psi);
template Array<complex_t> psi_vector(const PsiClassical& psi);
#endif // ENABLE_PSI_CLASSICAL

#ifdef ENABLE_PSI_EXACT
template void psi_vector(complex<double>* result, const PsiExact& psi);
template Array<complex_t> psi_vector(const PsiExact& psi);
#endif // ENABLE_PSI_EXACT

} // namespace ann_on_gpu


#endif // ENABLE_EXACT_SUMMATION
