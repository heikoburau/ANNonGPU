#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiNorm.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
double psi_norm(const Psi_t& psi, Ensemble& exact_summation) {

    Array<double> result_ar(1, exact_summation.gpu);
    result_ar.clear();

    auto psi_kernel = psi.kernel();
    auto result_ptr = result_ar.data();

    exact_summation.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int spin_index,
            typename Ensemble::Basis_t& conf,
            const complex_t log_psi,
            typename Psi_t::Payload& payload,
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


#ifdef ENABLE_SPINS
template double psi_norm(const PsiDeep& psi, ExactSummationSpins&);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template double psi_norm(const PsiDeep& psi, ExactSummationPaulis&);
#endif // ENABLE_PAULIS


} // namespace ann_on_gpu

#endif // ENABLE_EXACT_SUMMATION
