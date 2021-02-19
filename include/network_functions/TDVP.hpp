#pragma once

#include "operators.hpp"
#include "Array.hpp"
#include "types.h"

#include <memory>


namespace ann_on_gpu {


using namespace std;


struct TDVP {
    const unsigned int num_params;

    Array<complex_t> E_local;
    Array<double> E2_local;
    Array<complex_t> O_k_ar;

    Array<complex_t> S_matrix;
    Array<complex_t> F_vector;

    Array<double>    prob_ratio;

    unique_ptr<Array<complex_t>> O_k_samples;
    unique_ptr<Array<double>>    weight_samples;

    inline TDVP(unsigned int num_params, bool gpu)
    :
    num_params(num_params),
    E_local(1, gpu),
    E2_local(1, gpu),
    O_k_ar(num_params, gpu),
    S_matrix(num_params * num_params, gpu),
    F_vector(num_params, gpu),
    prob_ratio(1, gpu)
    {}

    template<typename Psi_t, typename Ensemble>
    void eval(const Operator_t& op, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    void eval_with_psi_ref(const Operator_t& op, Psi_t& psi, Ensemble& ensemble);

    inline double var_H() const {
        return this->E2_local.front() - abs2(E_local.front());
    }
};


} // namespace ann_on_gpu
