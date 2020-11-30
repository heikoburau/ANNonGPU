#pragma once

#include "operator/Operator.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
Array<complex_t> apply_operator(Psi_t& psi, const Operator& op, Ensemble& ensemble);


} // namespace ann_on_gpu
