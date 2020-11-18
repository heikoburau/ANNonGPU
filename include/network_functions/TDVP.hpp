#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "types.h"


namespace ann_on_gpu {


struct TDVP {
    const unsigned int num_params;

    Array<complex_t> E_local;
    Array<double> E_local2;
    Array<complex_t> O_k_ar;

    Array<complex_t> S_matrix;
    Array<complex_t> F_vector;

    Array<double>    prob_ratio;

    inline TDVP(unsigned int num_params, bool gpu)
    :
    num_params(num_params),
    E_local(1, gpu),
    E_local2(1, gpu),
    O_k_ar(num_params, gpu),
    S_matrix(num_params * num_params, gpu),
    F_vector(num_params, gpu),
    prob_ratio(1, gpu)
    {}

    template<typename Psi_t, typename Ensemble>
    void eval(const Operator& op, Psi_t& psi, Ensemble& ensemble);

    double var_H() const;
};


} // namespace ann_on_gpu
