#pragma once

#include "operator/Operator.hpp"
#include "bases.hpp"

#include "quantum_state/PsiFullyPolarized.hpp"
#include "quantum_state/PsiDeep.hpp"

#include "cuda_complex.hpp"
#include "Array.hpp"
#include "types.h"


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


template<typename dtype_t, unsigned int order, typename PsiRef>
struct PsiClassical_t {

    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_local_energies = 2 * MAX_SPINS;

    struct Payload {
        // todo: add configuration to detect whether re-calculation is neccessary at all.

        dtype       log_psi_ref;

        complex_t   local_energies[max_local_energies];

        typename PsiRef::Payload ref_payload;
    };

    dtype*          params;
    unsigned int    num_params;
    unsigned int    num_sites;

    // data for the second moment
    struct M_2 {
        // Hamiltonian^2, but having only terms which are not congruent to each other (symmetric).
        Operator        symmetric_terms;

        unsigned int    begin;
        unsigned int    end;
    };

    // data for the squared first moment
    struct M_1_squared {
        // Hamiltonian, complete.
        // (symmetric, pure-x Pauli-strings may be replaced by a single representative,
        //  for the Spin basis, since PsiRef is symmetric.)
        Operator        terms;

        // N * (N + 1) / 2 index pairs for evaluating the double-sum of (m_1)^2.
        unsigned int    num_pairs;
        unsigned int*   ids_i;
        unsigned int*   ids_j;

        unsigned int    begin;
        unsigned int    end;
    };

    double       prefactor;

    // local Hamiltonian.
    Operator     H_local;

    M_2          m_2;
    M_1_squared  m_1_squared;

    PsiRef       psi_ref;


#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration) const {
        this->psi_ref.init_payload(payload.ref_payload, configuration);
    }

    template<typename Basis_t>
    HDINLINE
    void compute_local_energies(const Basis_t& configuration, Payload& payload) const {
        this->compute_1st_order_local_energies(configuration, payload);

        if(order > 1u) {
            this->compute_2nd_order_local_energies(configuration, payload);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void compute_1st_order_local_energies(const Basis_t& configuration, Payload& payload) const {
        SHARED_MEM_LOOP_BEGIN(n, this->H_local.num_strings) {
            this->H_local.nth_local_energy_symmetric(
                payload.local_energies[n],
                n,
                this->psi_ref,
                configuration,
                payload.log_psi_ref,
                payload.ref_payload
            );

            SHARED_MEM_LOOP_END(n);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void compute_2nd_order_local_energies(const Basis_t& configuration, Payload& payload) const {
        SHARED_MEM_LOOP_BEGIN(n, this->m_2.symmetric_terms.num_strings) {
            this->m_2.symmetric_terms.nth_local_energy_symmetric(
                payload.local_energies[this->m_2.begin + n],
                n,
                this->psi_ref,
                configuration,
                payload.log_psi_ref,
                payload.ref_payload
            );

            SHARED_MEM_LOOP_END(n);
        }

        SHARED_MEM_LOOP_BEGIN(m, this->m_1_squared.terms.num_strings) {
            SINGLE {
                payload.local_energies[this->m_1_squared.begin + m] = complex_t(0.0);
            }

            this->m_1_squared.terms.nth_local_energy(
                payload.local_energies[this->m_1_squared.begin + m],
                m,
                this->psi_ref,
                configuration,
                payload.log_psi_ref,
                payload.ref_payload
            );

            SHARED_MEM_LOOP_END(m);
        }
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        this->psi_ref.log_psi_s(payload.log_psi_ref, configuration, payload.ref_payload);

        SINGLE {
            result = payload.log_psi_ref;
        }

        this->compute_local_energies(configuration, payload);

        LOOP(k, this->num_params) {
            generic_atomicAdd(&result, this->params[k] * this->get_O_k(k, payload));
        }
    }

    template<typename Basis_t>
    HDINLINE
    dtype psi_s(const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, configuration, payload);

        return exp(log(this->prefactor) + log_psi);
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {
        this->psi_ref.update_input_units(old_vector, new_vector, payload.ref_payload);
    }

    HDINLINE
    complex_t get_O_k(const unsigned int k, Payload& payload) const {
        if(k < this->H_local.num_strings) {
            return payload.local_energies[k];
        }

        if(order > 1u) {
            if(k < this->m_2.end) {
                return payload.local_energies[k];
            }

            if(k < this->m_1_squared.end) {
                return (
                    payload.local_energies[
                        this->m_1_squared.begin + this->m_1_squared.ids_i[k - this->m_1_squared.begin]
                    ] *
                    payload.local_energies[
                        this->m_1_squared.begin + this->m_1_squared.ids_j[k - this->m_1_squared.begin]
                    ]
                );
            }
        }

        return complex_t(0.0);
    }

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        this->psi_ref.log_psi_s(payload.log_psi_ref, configuration, payload.ref_payload);

        this->compute_local_energies(configuration, payload);

        LOOP(k, this->num_params) {
            function(k, this->get_O_k(k, payload));
        }
    }

    PsiClassical_t& kernel() {
        return *this;
    }

    const PsiClassical_t& kernel() const {
        return *this;
    }

    #endif // __CUDACC__

    HDINLINE
    unsigned int get_width() const {
        return this->psi_ref.get_width();
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->psi_ref.N;
    }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }
};


}  // namespace kernel



template<typename dtype, unsigned int order, typename PsiRef>
struct PsiClassical_t : public kernel::PsiClassical_t<dtype, order, typename PsiRef::Kernel> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    Operator&       H_local_op;
    Operator&       M_2_op;
    Operator&       M_1_squared_op;
    Array<dtype>    params;
    PsiRef          psi_ref;
    bool            gpu;

    Array<unsigned int> ids_i;
    Array<unsigned int> ids_j;

    inline PsiClassical_t(const PsiClassical_t& other)
        :
        H_local_op(other.H_local_op),
        M_2_op(other.M_2_op),
        M_1_squared_op(other.M_1_squared_op),
        params(other.params),
        psi_ref(other.psi_ref),
        ids_i(other.gpu),
        ids_j(other.gpu)
    {
        this->num_sites = other.num_sites;
        this->prefactor = other.prefactor;
        this->gpu = gpu;

        this->init_kernel();
    }

#ifdef __PYTHONCC__

    inline PsiClassical_t(
        const unsigned int num_sites,
        const Operator& H_local_op,
        const Operator& M_2_op,
        const Operator& M_1_squared_op,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& params,
        const PsiRef& psi_ref,
        const double prefactor,
        const bool gpu
    )
        :
        H_local_op(H_local_op),
        M_2_op(M_2_op),
        M_1_squared_op(M_1_squared_op),
        params(params, gpu),
        psi_ref(psi_ref),
        ids_i(gpu),
        ids_j(gpu)
    {
        this->num_sites = num_sites;
        this->prefactor = prefactor;
        this->gpu = gpu;

        this->init_kernel();
    }

    PsiClassical_t copy() const {
        return *this;
    }

#endif // __PYTHONCC__

    void init_kernel();

};

template<unsigned int order>
using PsiClassicalFP = PsiClassical_t<complex_t, order, PsiFullyPolarized>;

template<unsigned int order>
using PsiClassicalANN = PsiClassical_t<complex_t, order, PsiDeep>;

}  // namespace ann_on_gpu
