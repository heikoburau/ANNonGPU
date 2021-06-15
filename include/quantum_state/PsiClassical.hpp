#pragma once

#include "operators.hpp"
#include "bases.hpp"

#include "quantum_state/PsiFullyPolarized.hpp"
// #include "quantum_state/PsiCNN.hpp"
#include "quantum_state/PsiDeep.hpp"

#include "cuda_complex.hpp"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;

    #include "QuantumExpression/QuantumExpression.hpp"
#endif // __PYTHONCC__


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace PsiClassicalPayload {

template<typename PsiRefPayload>
struct Payload_t {
    complex_t       log_psi_ref;
    complex_t       local_energies[200];

    PsiRefPayload   ref_payload;
};

} // namespace PsiClassicalPayload

template<typename dtype_t, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
struct PsiClassical_t {

    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    using Payload = PsiClassicalPayload::Payload_t<typename PsiRef::Payload>;

    dtype*          params;
    unsigned int    num_params;
    unsigned int    num_sites;

    complex_t       log_prefactor;

    Operator_t*     H_local;

    unsigned int    num_ops_H;

    PsiRef       psi_ref;

#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        this->psi_ref.init_payload(payload.ref_payload, configuration, conf_idx);

        MULTI(n, this->num_ops_H) {
            this->H_local[n].fast_local_energy(
                payload.local_energies[n],
                configuration
            );
        }

        SYNC; // might not be neccessary
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = this->log_prefactor;
        }

        this->init_payload(payload, configuration, 0u);

        MULTI(k, this->num_ops_H) {
            generic_atomicAdd(&result, this->params[k] * payload.local_energies[k]);
        }

        if(order > 1u) {
            this->psi_ref.log_psi_s(payload.log_psi_ref, configuration, payload.ref_payload);

            SINGLE {
                result += payload.log_psi_ref;
            }
        }

        SYNC;
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {

    }

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        MULTI(k, this->num_ops_H) {
            function(k, payload.local_energies[k]);
        }

        if(order > 1u) {
            this->psi_ref.foreach_O_k(
                configuration,
                payload.ref_payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    function(this->num_ops_H + k, O_k);
                }
            );
        }
    }



#endif // __CUDACC__

    const PsiClassical_t& kernel() const {
        return *this;
    }

    PsiClassical_t& kernel() {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return max(this->num_ops_H, this->psi_ref.get_width());
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->num_sites;
    }
};


}  // namespace kernel



template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef_t>
struct PsiClassical_t : public kernel::PsiClassical_t<dtype, typename Operator_t::Kernel, order, symmetric, typename PsiRef_t::Kernel> {
    using PsiRef = PsiRef_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Operator = Operator_t;

    vector<Operator_t>        H_local;
    Array<typename Operator_t::Kernel>        H_local_kernel;
    Array<dtype>    params;
    PsiRef          psi_ref;
    bool            gpu;

    inline PsiClassical_t(const PsiClassical_t& other)
        :
        H_local(other.H_local),
        H_local_kernel(other.H_local_kernel),
        params(other.params),
        psi_ref(other.psi_ref)
    {
        this->num_sites = other.num_sites;
        this->log_prefactor = other.log_prefactor;
        this->gpu = other.gpu;

        this->init_kernel();
        this->update_kernel();
    }

    inline void update_psi_ref_kernel() {
        this->kernel().psi_ref = this->psi_ref.kernel();
    }

    inline void update_kernel() {
        this->update_psi_ref_kernel();
    }

    Array<dtype> get_params() const;
    void set_params(const Array<dtype>& new_params);

#ifdef __PYTHONCC__

    inline PsiClassical_t(
        const unsigned int num_sites,
        const vector<Operator_t>& H_local,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& params,
        const PsiRef& psi_ref,
        const std::complex<double> log_prefactor,
        const bool gpu
    )
        :
        H_local(H_local),
        H_local_kernel(H_local.size(), gpu),
        params(params, gpu),
        psi_ref(psi_ref)
    {
        this->num_sites = num_sites;
        this->log_prefactor = log_prefactor;
        this->gpu = gpu;

        this->init_kernel();
    }

    PsiClassical_t copy() const {
        return *this;
    }

    inline unsigned int get_order() const {
        return order;
    }

#endif // __PYTHONCC__

    void init_kernel();

};

#ifdef PSI_CLASSICAL_SYMMETRIC

template<unsigned int order>
using PsiClassicalFP = PsiClassical_t<complex_t, Operator_t, order, true, PsiFullyPolarized>;

template<unsigned int order>
using PsiClassicalANN = PsiClassical_t<complex_t, Operator_t, order, true, PsiDeep>;

#else

template<unsigned int order>
using PsiClassicalFP = PsiClassical_t<complex_t, Operator_t, order, false, PsiFullyPolarized>;

template<unsigned int order>
using PsiClassicalANN = PsiClassical_t<complex_t, Operator_t, order, false, PsiDeep>;

#endif // PSI_CLASSICAL_SYMMETRIC

}  // namespace ann_on_gpu
