#pragma once

#include "PsiDeep.hpp"

#include "RNGStates.hpp"
#include "bases.hpp"
#include "Array.hpp"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


// #define DIM 1


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


template<bool symmetric>
struct PsiDeepSigned_t {
    using dtype = complex_t;
    using real_dtype = double;
    using MyPsiDeep = PsiDeepT<double, symmetric>;

    static constexpr unsigned int max_N = MyPsiDeep::max_N;
    static constexpr unsigned int max_layers = MyPsiDeep::max_layers;
    static constexpr unsigned int max_width = MyPsiDeep::max_width;
    static constexpr unsigned int max_deep_angles = MyPsiDeep::max_deep_angles;


    struct Payload {
        typename MyPsiDeep::Payload plus, minus;
    };

    MyPsiDeep psi_plus;
    MyPsiDeep psi_minus;

    unsigned int   N;
    unsigned int   num_sites;

    unsigned int   num_params;

    double         prefactor;
    dtype          log_prefactor;

#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration) const {
        if(!symmetric) {
            this->psi_plus.compute_angles(payload.plus.angles, configuration);
            this->psi_minus.compute_angles(payload.minus.angles, configuration);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s(complex_t& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SHARED double log_psi_plus;
        SHARED double log_psi_minus;

        this->psi_plus.log_psi_s(log_psi_plus, configuration, payload.plus);
        this->psi_minus.log_psi_s(log_psi_minus, configuration, payload.minus);

        SINGLE {
            // result = log(complex_t(exp(log_psi_plus) - exp(log_psi_minus)));
            result = log_psi_plus > log_psi_minus ? complex_t(log_psi_plus) : complex_t(log_psi_minus, 3.141592653589793);
            // result = complex_t(log_psi_plus);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void log_psi_s(double& result, const Basis_t& configuration, Payload& payload) const {
        SHARED complex_t log_psi;

        this->log_psi_s(log_psi, configuration, payload);

        SINGLE {
            result = log_psi.real();
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
        #include "cuda_kernel_defines.h"
        if(symmetric) {
            return;
        }

        this->psi_plus.update_input_units(old_vector, new_vector, payload.plus);
        this->psi_minus.update_input_units(old_vector, new_vector, payload.minus);
    }

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, configuration, payload);
        SYNC;

        if(log_psi.imag() == 0.0) {
            this->psi_plus.foreach_O_k(configuration, payload.plus, function);
            return;
        }

        this->psi_minus.foreach_O_k(
            configuration,
            payload.minus,
            [&](const unsigned int k, const complex_t& O_k) {
                function(this->psi_plus.num_params + k, O_k);
            }
        );

    }


#endif // __CUDACC__

    const PsiDeepSigned_t& kernel() const {
        return *this;
    }

    PsiDeepSigned_t& kernel() {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->psi_plus.width;
    }


    HDINLINE unsigned int get_num_input_units() const {
        return this->N;
    }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

};


} // namespace kernel


template<bool symmetric_t>
struct PsiDeepSigned_t : public kernel::PsiDeepSigned_t<symmetric_t> {
    constexpr static bool symmetric = symmetric_t;

    using MyPsiDeep = PsiDeepT<double, symmetric>;
    using Kernel = kernel::PsiDeepSigned_t<symmetric>;

    MyPsiDeep    psi_plus;
    MyPsiDeep    psi_minus;
    bool         gpu;

    PsiDeepSigned_t(const PsiDeepSigned_t& other);
    PsiDeepSigned_t& operator=(const PsiDeepSigned_t& other);

#ifdef __PYTHONCC__

    inline PsiDeepSigned_t(
        const MyPsiDeep& psi_plus,
        const MyPsiDeep& psi_minus
    ) : psi_plus(psi_plus), psi_minus(psi_minus), gpu(psi_plus.gpu) {
        this->N = psi_plus.N;
        this->num_sites = psi_plus.num_sites;

        this->num_params = psi_plus.num_params + psi_minus.num_params;

        this->prefactor = psi_plus.prefactor;
        this->log_prefactor = psi_plus.log_prefactor;

        this->init_kernel();
    }

    inline PsiDeepSigned_t copy() const {
        return *this;
    }

#endif // __PYTHONCC__

    Array<double> get_params() const;
    void set_params(const Array<double>& new_params);

    inline bool is_symmetric() const {
        return symmetric;
    }

    void init_kernel();
    void update_kernel();

};


#ifdef PSI_DEEP_SYMMETRIC
using PsiDeepSigned = PsiDeepSigned_t<true>;
#else
using PsiDeepSigned = PsiDeepSigned_t<false>;
#endif // PSI_DEEP_SYMMETRIC


} // namespace ann_on_gpu

