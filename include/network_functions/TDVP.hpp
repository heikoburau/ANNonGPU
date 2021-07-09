#pragma once

#include "operators.hpp"
#include "Array.hpp"
#include "types.h"

#include <memory>


namespace ann_on_gpu {


using namespace std;


struct TDVP {
    bool gpu;
    const unsigned int num_params;

    Array<complex_t> E_local;
    Array<double> E2_local;
    Array<complex_t> O_k_ar;

    Array<complex_t> S_matrix;
    Array<complex_t> F_vector;
    Array<complex_t> input_vector;
    Array<complex_t> output_vector;

    unique_ptr<Array<complex_t>> O_k_samples;
    unique_ptr<Array<complex_t>> E_local_samples;
    unique_ptr<Array<double>>    weight_samples;

    double threshold;
    Array<double> total_weight;

    inline TDVP(unsigned int num_params, bool gpu)
    :
    gpu(gpu),
    num_params(num_params),
    E_local(1, gpu),
    E2_local(1, gpu),
    O_k_ar(num_params, gpu),
    S_matrix(num_params * num_params, gpu),
    F_vector(num_params, gpu),
    input_vector(num_params, gpu),
    output_vector(num_params, gpu),
    threshold(-1e6),
    total_weight(1, gpu)
    {}

    template<typename Psi_t, typename Ensemble, typename use_psi_ref>
    void eval(const Operator& op, Psi_t& psi, Ensemble& ensemble, use_psi_ref);

    template<typename Psi_t, typename Ensemble>
    void eval_F_vector(const Operator& op, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    void eval_fast(const Operator& op, Psi_t& psi, Ensemble& ensemble);

    template<typename Psi_t, typename Ensemble>
    void compute_averages(const Operator& op, Psi_t& psi, Ensemble& ensemble, true_t);

    template<typename Psi_t, typename Ensemble>
    void compute_averages(const Operator& op, Psi_t& psi, Ensemble& ensemble, false_t);

    template<typename Psi_t, typename Ensemble>
    void compute_averages_fast(const Operator& op, Psi_t& psi, Ensemble& ensemble);

    inline double var_H() const {
        return this->E2_local.front() - abs2(E_local.front());
    }

    template<typename Ensemble>
    void S_dot_vector(
        Ensemble& ensemble
    );

    #ifdef __PYTHONCC__

    template<typename Psi_t, typename Ensemble>
    inline void eval_py(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
        return this->eval(op, psi, ensemble, false_t());
    }

    template<typename Psi_t, typename Ensemble>
    inline void eval_F_vector_py(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
        return this->eval_F_vector(op, psi, ensemble);
    }

    template<typename Psi_t, typename Ensemble>
    inline void eval_with_psi_ref_py(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
        return this->eval(op, psi, ensemble, true_t());
    }

    template<typename Psi_t, typename Ensemble>
    inline void eval_fast_py(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
        return this->eval_fast(op, psi, ensemble);
    }

    #endif // __PYTHONCC__
};


} // namespace ann_on_gpu
