#pragma once

#ifdef ENABLE_EXACT_SUMMATION

#include "ensembles/ExactSummation.hpp"


namespace ann_on_gpu {

template<typename Psi_t, typename Ensemble>
double psi_norm(const Psi_t& psi, Ensemble& exact_summation);

} // namespace ann_on_gpu

#endif // ENABLE_EXACT_SUMMATION
