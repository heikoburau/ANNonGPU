#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

namespace ann_on_gpu {

using namespace std;


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_angles(const Psi_t& psi, Ensemble& spin_ensemble);

} // namespace ann_on_gpu
