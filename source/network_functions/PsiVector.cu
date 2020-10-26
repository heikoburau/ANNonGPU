#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiVector.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(const Psi_t& psi, Ensemble& ensemble) {

    Array<complex_t> result(ensemble.get_num_steps(), psi.gpu);

    auto result_ptr = result.data();
    auto log_prefactor = log(psi.prefactor);
    auto psi_kernel = psi.kernel();

    ensemble.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int conf_index,
            const typename Ensemble::Basis_t& basis_vector,
            const complex_t log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SINGLE {
                result_ptr[conf_index] = exp(log_prefactor + log_psi);
            }
        }
    );
    result.update_host();

    return result;
}


#ifdef ENABLE_SPINS
template Array<complex_t> psi_vector(const PsiDeep& psi, ExactSummationSpins& ensemble);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template Array<complex_t> psi_vector(const PsiDeep& psi, ExactSummationPaulis& ensemble);
#endif // ENABLE_PAULIS


} // namespace ann_on_gpu


#endif // ENABLE_EXACT_SUMMATION
