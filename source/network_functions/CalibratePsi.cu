// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// CalibratePsi.cu.template
// ***********************************************************

#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_CalibratePsi_cpp

#include "network_functions/CalibratePsi.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "ensembles.hpp"


namespace ann_on_gpu {

#ifdef ENABLE_EXACT_SUMMATION
template<typename Basis_t>
void calibrate(PsiDeep& psi, ExactSummation_t<Basis_t>& ensemble) {
    psi.prefactor = 1.0;
    psi.log_prefactor = complex_t(0.0);
    psi.prefactor /= psi_norm(psi, ensemble);
    psi.log_prefactor = -log_psi(psi, ensemble);
    psi.prefactor /= psi_norm(psi, ensemble);
}
#endif // ENABLE_EXACT_SUMMATION

#ifdef ENABLE_MONTE_CARLO
template<typename Basis_t>
void calibrate(PsiDeep& psi, MonteCarlo_tt<Basis_t>& ensemble) {
    psi.log_prefactor = complex_t(0.0);
    psi.log_prefactor = -log_psi(psi, ensemble);
}
#endif // ENABLE_MONTE_CARLO


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template void calibrate(PsiDeep&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
template void calibrate(PsiDeep&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template void calibrate(PsiDeep&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
template void calibrate(PsiDeep&, ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu
