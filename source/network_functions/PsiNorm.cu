// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// PsiNorm.cu.template
// ***********************************************************

#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiNorm.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
double psi_norm(Psi_t& psi, Ensemble& exact_summation) {

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


#if defined(ENABLE_SPINS)
template double psi_norm(PsiDeep& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS)
template double psi_norm(PsiDeepSigned& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double psi_norm(PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double psi_norm(PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_PAULIS)
template double psi_norm(PsiDeep& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS)
template double psi_norm(PsiDeepSigned& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template double psi_norm(PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double psi_norm(PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double psi_norm(PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu

#endif // ENABLE_EXACT_SUMMATION
