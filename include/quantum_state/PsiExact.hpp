#pragma once

#include "bases.hpp"
#include "Array.hpp"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __PYTHONCC__


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;


template<typename dtype_t>
struct PsiExact_t {
    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    struct Payload {
        unsigned int conf_idx;
    };


    unsigned int num_sites;
    dtype* vector;
    complex_t log_prefactor; // dummy



    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        payload.conf_idx = conf_idx;
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        SINGLE {
            result = this->vector[payload.conf_idx];
        }
        SYNC;
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {}

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {}

    const PsiExact_t& kernel() const {
        return *this;
    }

    PsiExact_t& kernel() {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->num_sites;
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->num_sites;
    }

};

} // namespace kernel

template<typename dtype_t>
struct PsiExact_t : public kernel::PsiExact_t<dtype_t> {
    using Kernel = kernel::PsiExact_t<dtype_t>;

    Array<dtype_t> vector;
    bool gpu;

    inline PsiExact_t(const PsiExact_t& other) : vector(other.vector), gpu(other.gpu) {
        this->num_sites = other.num_sites;
        this->vector.update_device();
        this->kernel().vector = this->vector.data();
    }

#ifdef __PYTHONCC__

    inline PsiExact_t(
        const xt::pytensor<typename std_dtype<dtype_t>::type, 1u>& py_vector,
        bool gpu
    ) : vector(py_vector, gpu), gpu(gpu) {

        this->num_sites = static_cast<unsigned int>(log2(this->vector.size()));

        this->vector.update_device();
        this->kernel().vector = this->vector.data();
    }

#endif // __PYTHONCC__
};


using PsiExact = PsiExact_t<complex_t>;

} // namespace ann_on_gpu
