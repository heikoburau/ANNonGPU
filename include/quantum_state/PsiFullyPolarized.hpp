#pragma once

#include "bases.hpp"

#include "cuda_complex.hpp"
#include "Array.hpp"
#include "types.h"


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;


template<typename dtype_t>
struct PsiFullyPolarized_t {

    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_angles = 0u;

    struct Payload {};

    unsigned int   num_sites;
    unsigned int   num_params;
    double         prefactor;
    bool           gpu;

#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload&, const Basis_t&) const {}

    HDINLINE
    void save_payload(Payload& payload) const {}

#ifdef ENABLE_SPINS
    template<typename result_dtype>
    HDINLINE
    void log_psi_s(result_dtype& result, const Spins& spins, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = result_dtype(0.0);
        }
    }
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
    template<typename result_dtype>
    HDINLINE
    void log_psi_s(result_dtype& result, const PauliString& paulis, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = paulis.has_no_sigma_yz() ? result_dtype(0.0) : complex_t(-6.0);
        }
    }
#endif // ENABLE_PAULIS

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
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {}

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
    }

#endif // __CUDACC__

    PsiFullyPolarized_t kernel() const {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->num_sites;
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->num_sites;
    }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * log_psi_s_real);
    }
};



}  // namespace kernel



template<typename dtype>
struct PsiFullyPolarized_t : public kernel::PsiFullyPolarized_t<dtype> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Kernel = kernel::PsiFullyPolarized_t<dtype>;

    inline PsiFullyPolarized_t(const PsiFullyPolarized_t& other)
    {
        this->num_sites = other.num_sites;
        this->prefactor = 1.0;
        this->num_params = 0u;
        this->gpu = false;
    }

#ifdef __PYTHONCC__

    inline PsiFullyPolarized_t(unsigned int num_sites) {
        this->num_sites = num_sites;
        this->prefactor = 1.0;
        this->num_params = 0u;
        this->gpu = false;
    }

    PsiFullyPolarized_t copy() const {
        return *this;
    }

#endif // __PYTHONCC__

    template<typename Ensemble>
    inline void calibrate(Ensemble& ensemble) {}

    inline void prepare(const unsigned int num_configurations) {}
};


using PsiFullyPolarized = PsiFullyPolarized_t<complex_t>;

}  // namespace ann_on_gpu
