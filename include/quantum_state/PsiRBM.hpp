#pragma once

#include "psi_functions.hpp"

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



namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


template<typename dtype_t, bool symmetric>
struct PsiRBM_t {
    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_N = 128u;
    static constexpr unsigned int max_width = max_N;

    template<typename dtype, unsigned int max_width>
    struct Payload_t {
        dtype angles[max_width];
    };

    using Payload = Payload_t<dtype, max_width>;

    unsigned int   N;
    unsigned int   num_sites;
    unsigned int   M;

    unsigned int   num_params;

    dtype          log_prefactor;

    dtype*         RESTRICT W;
    dtype          final_weight;

#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void compute_angles(dtype* angles, const Basis_t& configuration) const {
        MULTI(j, this->M) {
            angles[j] = dtype(0.0);

            for(auto i = 0u; i < this->N; i++) {
                angles[j] += this->W[i * this->M + j] * configuration[i];
            }
        }
        SYNC;
    }

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        if(!symmetric) {
            this->compute_angles(payload.angles, configuration);
        }
    }

    template<typename result_dtype>
    HDINLINE
    void forward_pass(
        result_dtype& result,
        dtype* RESTRICT angles
    ) const {
        #include "cuda_kernel_defines.h"

        dtype REGISTER(activation, this->max_width);

        MULTI(j, this->M) {
            REGISTER(activation, j) = my_logcosh(angles[j], 0u);

            generic_atomicAdd(&result, REGISTER(activation, j) * this->final_weight);
        }
        SYNC;
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = result_dtype(this->log_prefactor);
        }

        this->forward_pass(result, payload.angles);
    }

#if defined(ENABLE_SPINS) || defined(ENABLE_FERMIONS)
    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {
        // CAUTION: This functions assumes that the first hidden layer is fully connected to the input units!

        #include "cuda_kernel_defines.h"

        if(symmetric) {
            return;
        }

        // 'updated_units' must be shared
        SHARED uint64_t     updated_units;
        SHARED unsigned int unit_position;

        SINGLE {
            updated_units = old_vector.is_different(new_vector);
            unit_position = first_bit_set(updated_units) - 1u;
        }
        SYNC;

        while(updated_units) {
            MULTI(j, this->M) {
                payload.angles[j] += (
                    2.0 * new_vector[unit_position] * this->W[unit_position * this->M + j]
                );
            }
            SYNC;
            SINGLE {
                updated_units &= ~(1lu << unit_position);
                unit_position = first_bit_set(updated_units) - 1u;
            }
            SYNC;
        }
    }

#endif

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        dtype REGISTER(activation, this->max_width);

        MULTI(j, this->M) {
            REGISTER(activation, j) = this->final_weight * my_tanh(payload.angles[j], 0);

            for(auto i = 0u; i < this->N; i++) {
                function(
                    i * this->M + j,
                    REGISTER(activation, j) * configuration[i]
                );
            }
        }
    }

#endif // __CUDACC__

    const PsiRBM_t& kernel() const {
        return *this;
    }

    PsiRBM_t& kernel() {
        return *this;
    }

    HDINLINE unsigned int get_width() const {
        return this->M;
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->N;
    }
};

} // namespace kernel


template<typename dtype, bool symmetric>
struct PsiRBM_t : public kernel::PsiRBM_t<dtype, symmetric> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Kernel = kernel::PsiRBM_t<dtype, symmetric>;

    Array<dtype> W;
    bool         gpu;

    inline PsiRBM_t(const PsiRBM_t& other):
        W(other.W), gpu(other.gpu)
    {
        this->N = other.N;
        this->M = other.M;
        this->num_sites = this->N;
        this->num_params = this->N * this->M;

        this->final_weight = other.final_weight;
        this->log_prefactor = other.log_prefactor;

        this->kernel().W = this->W.data();
    }

#ifdef __PYTHONCC__

    inline PsiRBM_t
(
        const xt::pytensor<std::complex<double>, 2u>& W,
        const std::complex<double> final_weight,
        const std::complex<double> log_prefactor,
        const bool gpu
    ) : W(W, gpu), gpu(gpu)
    {

        this->N = W.shape()[0];
        this->M = W.shape()[1];
        this->num_sites = this->N;
        this->num_params = this->N * this->M;

        this->final_weight = complex_t(final_weight);
        this->log_prefactor = complex_t(log_prefactor);

        this->kernel().W = this->W.data();
    }

#endif // __PYTHONCC__

    PsiRBM_t copy() const {
        return *this;
    }

    inline Array<dtype> get_params() const {
        return this->W;
    }
    inline void set_params(const Array<dtype>& new_params) {
        this->W = new_params;
        this->W.update_device();
    }

    inline bool is_symmetric() const {
        return symmetric;
    }
};


#ifdef PSI_DEEP_SYMMETRIC
using PsiRBM = PsiRBM_t<complex_t, true>;
#else
using PsiRBM = PsiRBM_t<complex_t, false>;
#endif // PSI_DEEP_SYMMETRIC

// using PsiRBM = PsiRBM<cuda_complex::complex<double>>;

} // namespace ann_on_gpu
