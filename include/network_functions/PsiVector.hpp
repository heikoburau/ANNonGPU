#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif

namespace ann_on_gpu {

using namespace std;


template<typename Psi_t, typename Basis_t>
typename std_dtype<typename Psi_t::dtype>::type log_psi_s(Psi_t& psi, const Basis_t& configuration);

template<typename Psi_t, typename Ensemble>
typename std_dtype<typename Psi_t::dtype>::type log_psi(Psi_t& psi, Ensemble& ensemble);

template<typename Psi_t, typename Ensemble>
Array<complex_t> log_psi_vector(Psi_t& psi, Ensemble& ensemble);

template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(Psi_t& psi, Ensemble& ensemble);


} // namespace ann_on_gpu
