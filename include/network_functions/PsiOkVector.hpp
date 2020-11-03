#pragma once

#include "Array.hpp"
#include <complex>


namespace ann_on_gpu {


template<typename Psi_t, typename Basis_t>
Array<complex_t> psi_O_k(const Psi_t& psi, const Basis_t& configuration);


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_O_k_vector(const Psi_t& psi, Ensemble& ensemble);


} // namespace ann_on_gpu
