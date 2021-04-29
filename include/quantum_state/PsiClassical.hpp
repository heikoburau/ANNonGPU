#pragma once

#include "operators.hpp"
#include "bases.hpp"

#include "quantum_state/PsiFullyPolarized.hpp"
// #include "quantum_state/PsiCNN.hpp"
#include "quantum_state/PsiExact.hpp"

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

template<bool, typename PsiRefPayload, int max_local_terms, int max_sites>
struct Payload_t;

template<typename PsiRefPayload, int max_local_terms, int max_sites>
struct Payload_t<true, PsiRefPayload, max_local_terms, max_sites> {
    // todo: add configuration to detect whether re-calculation is neccessary at all.

    complex_t       log_psi_ref;

    // structure:
    //
    // first-order:
    // [this->num_sites] x [this->num_ops_H]
    //
    // second-order:
    // H^2_local terms
    //
    complex_t   local_energies[max_local_terms * max_sites];

    complex_t   local_energy_H_full[max_local_terms];

    PsiRefPayload ref_payload;
};

template<typename PsiRefPayload, int max_local_terms, int max_sites>
struct Payload_t<false, PsiRefPayload, max_local_terms, max_sites> {
    complex_t       log_psi_ref;

    complex_t   local_energies[max_local_terms * max_sites * (max_sites + 1)];
    complex_t   local_energy_H_full[1];

    PsiRefPayload ref_payload;
};

} // namespace PsiClassicalPayload

template<typename dtype_t, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
struct PsiClassical_t {

    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_sites = 32u;
    static constexpr unsigned int max_local_terms = 2u;

    using Payload = PsiClassicalPayload::Payload_t<symmetric, typename PsiRef::Payload, max_local_terms, max_sites>;

    dtype*          params;
    unsigned int    num_params;
    unsigned int    num_sites;

    // data for the second moment
    struct M_2 {
        // Hamiltonian^2, but having only terms which are not congruent to each other (symmetric).

        unsigned int    begin_local_energies;
        unsigned int    begin_params;
        unsigned int    end_params;
    };

    // data for the squared first moment
    struct M_1_squared {
        // N * (N + 1) / 2 index pairs for evaluating the double-sum of (m_1)^2.
        unsigned int    num_ll_pairs;
        unsigned int*   ids_l;
        unsigned int*   ids_l_prime;

        unsigned int    begin_params;
    };

    complex_t       log_prefactor;

    Operator_t*     H_local;
    Operator_t*     H_2_local;

    unsigned int    num_ops_H;
    unsigned int    num_ops_H_2;

    M_2          m_2;
    M_1_squared  m_1_squared;

    PsiRef       psi_ref;

    double       log_psi_threshold;
    Operator_t   H;
    Operator_t   H2;
    double       delta_t;


#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        this->psi_ref.init_payload(payload.ref_payload, configuration, conf_idx);
        this->psi_ref.log_psi_s(payload.log_psi_ref, configuration, payload.ref_payload);

        this->compute_local_energies(configuration, payload);
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
        SHARED_MEM_LOOP_BEGIN(n, this->num_ops_H) {

            if(symmetric) {

                SINGLE {
                    payload.local_energy_H_full[n] = complex_t(0.0);
                }

                SHARED_MEM_LOOP_BEGIN(m, this->num_sites) {
                    this->H_local[n].local_energy(
                        payload.local_energies[m * this->num_ops_H + n],
                        this->psi_ref,
                        configuration,
                        payload.log_psi_ref,
                        payload.ref_payload,
                        m,
                        true
                    );

                    SHARED_MEM_LOOP_END(m);
                }
                MULTI(i, this->num_sites) {
                    generic_atomicAdd(
                        &payload.local_energy_H_full[n],
                        payload.local_energies[i * this->num_ops_H + n]
                    );
                }

            }
            else {
                this->H_local[n].local_energy(
                    payload.local_energies[n],
                    this->psi_ref,
                    configuration,
                    payload.log_psi_ref,
                    payload.ref_payload,
                    0,
                    true
                );
            }

            SHARED_MEM_LOOP_END(n);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void compute_2nd_order_local_energies(const Basis_t& configuration, Payload& payload) const {
        SHARED_MEM_LOOP_BEGIN(n, this->num_ops_H_2) {

            if(symmetric) {
                SINGLE {
                    payload.local_energies[this->m_2.begin_local_energies + n] = complex_t(0.0);
                }

                SHARED_MEM_LOOP_BEGIN(m, this->num_sites) {
                    this->H_2_local[n].local_energy(
                        payload.local_energies[this->m_2.begin_local_energies + n],
                        this->psi_ref,
                        configuration,
                        payload.log_psi_ref,
                        payload.ref_payload,
                        m,
                        false
                    );

                    SHARED_MEM_LOOP_END(m);
                }
            }
            else {
                this->H_2_local[n].local_energy(
                    payload.local_energies[this->m_2.begin_local_energies + n],
                    this->psi_ref,
                    configuration,
                    payload.log_psi_ref,
                    payload.ref_payload,
                    0,
                    true
                );
            }

            SHARED_MEM_LOOP_END(n);
        }
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

        this->foreach_O_k(
            configuration,
            payload,
            [&](const unsigned int k, const complex_t& O_k) {
                generic_atomicAdd(&result, this->params[k] * O_k);
            }
        );

        // SHARED complex_t local_energy_H;
        // SHARED complex_t local_energy_H2;

        // this->H.local_energy(local_energy_H, this->psi_ref, configuration, payload.log_psi_ref, payload.ref_payload);
        // this->H2.local_energy(local_energy_H2, this->psi_ref, configuration, payload.log_psi_ref, payload.ref_payload);

        // SYNC;
        // if(payload.log_psi_ref.real() < this->log_psi_threshold) {
        //     SINGLE {
        //         result = this->log_prefactor + payload.log_psi_ref + log(
        //             1.0 -
        //             complex_t(0.0, 1.0) * local_energy_H * this->delta_t -
        //             0.5 * local_energy_H2 * this->delta_t * this->delta_t
        //         );
        //     }
        //     SYNC;

        //     return;
        // }

        // SINGLE {
        //     result = this->log_prefactor + payload.log_psi_ref + (
        //         -complex_t(0.0, 1.0) * local_energy_H * this->delta_t - 0.5 * (
        //             local_energy_H2 - local_energy_H * local_energy_H
        //         ) * this->delta_t * this->delta_t
        //     );
        // }
        SYNC;
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {

    }

    HDINLINE
    complex_t get_O_k(const unsigned int k, const Payload& payload) const {
        if(k < this->num_ops_H) {
            if(symmetric) {
                return payload.local_energy_H_full[k];
            }
            else {
                return payload.local_energies[k];
            }
        }

        if(order > 1u) {
            if(k < this->m_2.end_params) {
                return payload.local_energies[this->m_2.begin_local_energies + k - this->m_2.begin_params];
            }

            if(k < this->num_params) {
                complex_t result(0.0);

                const auto k_rel = k - this->m_1_squared.begin_params;

                const auto n = k_rel / this->m_1_squared.num_ll_pairs;
                const auto l = this->m_1_squared.ids_l[k_rel % this->m_1_squared.num_ll_pairs];
                const auto l_prime = this->m_1_squared.ids_l_prime[k_rel % this->m_1_squared.num_ll_pairs];

                if(symmetric)
                {
                    for(auto delta = 0u; delta < this->num_sites; delta++) {
                        result += (
                            payload.local_energies[delta * this->num_ops_H + l] *
                            payload.local_energies[((n + delta) % this->num_sites) * this->num_ops_H + l_prime]
                        );
                    }
                }
                else {
                    result = (
                        payload.local_energies[l] *
                        payload.local_energies[l_prime]
                    );
                }

                return result;
            }
        }

        return complex_t(0.0);
    }

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        LOOP(k, this->num_params) {
            function(k, this->get_O_k(k, payload));
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
        return this->psi_ref.get_width();
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->psi_ref.get_num_input_units();
    }
};


}  // namespace kernel



template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef_t>
struct PsiClassical_t : public kernel::PsiClassical_t<dtype, typename Operator_t::Kernel, order, symmetric, typename PsiRef_t::Kernel> {
    using PsiRef = PsiRef_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Operator = Operator_t;

    vector<Operator_t>        H_local;
    vector<Operator_t>        H_2_local;
    Array<typename Operator_t::Kernel>        H_local_kernel;
    Array<typename Operator_t::Kernel>        H_2_local_kernel;
    Array<dtype>    params;
    PsiRef          psi_ref;
    bool            gpu;

    Array<unsigned int> ids_l;
    Array<unsigned int> ids_l_prime;

    unique_ptr<Operator_t> H;
    unique_ptr<Operator_t> H2;

    inline PsiClassical_t(const PsiClassical_t& other)
        :
        H_local(other.H_local),
        H_2_local(other.H_2_local),
        H_local_kernel(other.H_local_kernel),
        H_2_local_kernel(other.H_2_local_kernel),
        params(other.params),
        psi_ref(other.psi_ref),
        ids_l(other.gpu),
        ids_l_prime(other.gpu)
    {
        this->num_sites = other.num_sites;
        this->log_prefactor = other.log_prefactor;
        this->gpu = other.gpu;
        this->delta_t = other.delta_t;
        this->log_psi_threshold = other.log_psi_threshold;

        if(other.H) {
            this->H = unique_ptr<Operator_t>(new Operator_t(*other.H));
        }
        if(other.H2) {
            this->H2 = unique_ptr<Operator_t>(new Operator_t(*other.H2));
        }

        this->init_kernel();
        this->update_kernel();
    }

    inline void update_psi_ref_kernel() {
        this->kernel().psi_ref = this->psi_ref.kernel();
    }

    inline void update_kernel() {
        if(this->H) {
            this->kernel().H = this->H->kernel();
        }
        if(this->H2) {
            this->kernel().H2 = this->H2->kernel();
        }
    }

#ifdef __PYTHONCC__

    inline PsiClassical_t(
        const unsigned int num_sites,
        const vector<Operator_t>& H_local,
        const vector<Operator_t>& H_2_local,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& params,
        const PsiRef& psi_ref,
        const double log_prefactor,
        const bool gpu
    )
        :
        H_local(H_local),
        H_2_local(H_2_local),
        H_local_kernel(H_local.size(), gpu),
        H_2_local_kernel(H_2_local.size(), gpu),
        params(params, gpu),
        psi_ref(psi_ref),
        ids_l(gpu),
        ids_l_prime(gpu)
    {
        this->num_sites = num_sites;
        this->log_prefactor = log_prefactor;
        this->gpu = gpu;
        this->log_psi_threshold = -1e6;

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
using PsiClassicalANN = PsiClassical_t<complex_t, Operator_t, order, true, PsiExact>;

#else

template<unsigned int order>
using PsiClassicalFP = PsiClassical_t<complex_t, Operator_t, order, false, PsiFullyPolarized>;

template<unsigned int order>
using PsiClassicalANN = PsiClassical_t<complex_t, Operator_t, order, false, PsiExact>;

#endif // PSI_CLASSICAL_SYMMETRIC

}  // namespace ann_on_gpu
