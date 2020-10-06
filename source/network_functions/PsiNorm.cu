#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiNorm.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t>
double psi_norm(const Psi_t& psi, ExactSummation& exact_summation) {

    Array<double> result_ar(1, exact_summation.gpu);
    result_ar.clear();

    auto psi_kernel = psi.kernel();
    auto result_ptr = result_ar.data();

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
            #include "cuda_kernel_defines.h"

            SINGLE {
                generic_atomicAdd(result_ptr, psi_kernel.probability_s(log_psi.real()));
            }
        }
    );

    result_ar.update_host();

    return sqrt(result_ar.front());
}


#ifdef ENABLE_PSI_DEEP
template double psi_norm(const PsiDeep& psi, ExactSummation&);
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_PAIR
template double psi_norm(const PsiPair& psi, ExactSummation&);
#endif // ENABLE_PSI_PAIR

#ifdef ENABLE_PSI_EXACT
template double psi_norm(const PsiExact& psi, ExactSummation&);
#endif // ENABLE_PSI_EXACT

} // namespace ann_on_gpu

#endif // ENABLE_EXACT_SUMMATION
