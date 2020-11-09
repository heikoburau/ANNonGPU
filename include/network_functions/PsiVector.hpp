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
std::complex<double> log_psi_s(const Psi_t& psi, const Basis_t& configuration);

template<typename Psi_t, typename Ensemble>
std::complex<double> log_psi(const Psi_t& psi, Ensemble& ensemble);


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(const Psi_t& psi, Ensemble& ensemble);


} // namespace ann_on_gpu
