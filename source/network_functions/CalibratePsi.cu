// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// CalibratePsi.cu.template
// ***********************************************************

// #define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_CalibratePsi_cpp

#include "network_functions/CalibratePsi.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"


namespace ann_on_gpu {

// #ifdef ENABLE_EXACT_SUMMATION
// template<typename Psi_t>
// void calibrate(Psi_t& psi, ExactSummation_t<Spins>& ensemble) {
//     psi.prefactor = 1.0;
//     psi.log_prefactor = complex_t(0.0);
//     psi.prefactor /= psi_norm(psi, ensemble);
//     psi.log_prefactor = -log_psi(psi, ensemble);
//     psi.prefactor /= psi_norm(psi, ensemble);
// }
// #endif // ENABLE_EXACT_SUMMATION

// #ifdef ENABLE_MONTE_CARLO
// template<typename Psi_t>
// void calibrate(Psi_t& psi, MonteCarlo_tt<Spins>& ensemble) {
//     psi.log_prefactor = complex_t(0.0);
//     psi.log_prefactor = -log_psi(psi, ensemble);
// }
// #endif // ENABLE_MONTE_CARLO




} // namespace ann_on_gpu
