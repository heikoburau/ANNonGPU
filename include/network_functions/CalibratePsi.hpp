#pragma once


#include "quantum_state/PsiDeep.hpp"


namespace ann_on_gpu {


template<typename Ensemble_t>
void calibrate(PsiDeep& psi, Ensemble_t& ensemble);


} // namespace ann_on_gpu
