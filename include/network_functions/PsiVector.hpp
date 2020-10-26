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


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(const Psi_t& psi, Ensemble& ensemble);


#ifdef __PYTHONCC__

template<typename Psi_t, typename Ensemble>
inline xt::pytensor<complex<double>, 1u> psi_vector_py(const Psi_t& psi, Ensemble& ensemble) {
    return psi_vector(psi, ensemble).to_pytensor_1d();
}

#endif

} // namespace ann_on_gpu
