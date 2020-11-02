#pragma once

#include "bases.hpp"

#include "cuda_complex.hpp"
#include "Array.hpp"
#include "types.h"


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;


template<typename dtype_t>
struct PsiProductState_t {

    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_angles = 0u;

    struct Payload {};

    unsigned int   N;
    dtype*         input_weights;

#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void compute_angles(dtype* angles, const Basis_t& configuration) const {}

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s_generic(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = result_dtype(0.0);
        }
        SYNC;
        MULTI(i, this->N) {
            generic_atomicAdd(&result, result_dtype(configuration.network_unit_at(i)) * this->input_weights[i]);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s(dtype& result, const Basis_t& configuration, Payload& payload) const {
        this->log_psi_s_generic(result, configuration, payload);
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s_real(real_dtype& result, const Basis_t& configuration, Payload& payload) const {
        this->log_psi_s_generic(result, configuration, payload);
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s(dtype& result, const Basis_t& configuration, dtype* angles, Payload& payload) const {
        this->log_psi_s_generic(result, configuration, payload);
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s_real(real_dtype& result, const Basis_t& configuration, dtype* angles, Payload& payload) const {
        this->log_psi_s_generic(result, configuration, payload);
    }

    template<typename Basis_t>
    HDINLINE
    dtype psi_s(const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, configuration, payload);

        return exp(log_psi);
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        dtype* angles, const Basis_t& old_vector, const Basis_t& new_vector
    ) const {}

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        MULTI(i, this->N) {
            function(
                i,
                get_real<dtype>(static_cast<real_dtype>(
                    configuration.network_unit_at(i)
                ))
            );
        }
    }

    PsiProductState_t kernel() const {
        return *this;
    }

    #endif // __CUDACC__
};



}  // namespace kernel



template<typename dtype>
struct PsiProductState_t : public kernel::PsiProductState_t<dtype> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    Array<dtype> input_weights;
    bool         gpu;

    inline PsiProductState_t(const PsiProductState_t& other)
        :
        input_weights(other.input_weights),
        gpu(gpu)
    {
        this->N = other.N;
        this->kernel().input_weights = this.input_weights.data();
    }

#ifdef __PYTHONCC__

    inline PsiProductState_t(
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& input_weights,
        const bool gpu
    ) : input_weights(input_weights, gpu){
        this->N = input_weights.shape()[0];
        this->gpu = gpu;

    PsiProductState_t copy() const {
        return *this;
    }

#endif // __PYTHONCC__

};


using PsiProductState = PsiProductState_t<complex_t>;

}  // namespace ann_on_gpu
