#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

namespace ann_on_gpu {

using namespace std;


template<typename Psi_t, typename SpinEnsemble>
Array<complex_t> psi_angles(const Psi_t& psi, SpinEnsemble& spin_ensemble);

} // namespace ann_on_gpu
