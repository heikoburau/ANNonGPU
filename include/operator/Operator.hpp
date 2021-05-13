#pragma once

#include "basis/PauliString.hpp"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include <xtensor-python/pytensor.hpp>
#endif // __CUDACC__

#ifdef __CUDACC__
    namespace quantum_expression {
        class PauliExpression;
    }
#else
    #include "QuantumExpression/QuantumExpression.hpp"
#endif


namespace ann_on_gpu {

namespace kernel {

struct Operator {
    complex_t*      coefficients;
    PauliString*    pauli_strings;
    unsigned int    num_strings;

#ifdef __CUDACC__

    template<typename Psi_t, typename Basis_t>
    HDINLINE
    void nth_local_energy(
        complex_t& result,
        unsigned int n,
        const Psi_t& psi,
        const Basis_t& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload,
        const unsigned int shift = 0u
    ) const {
        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;
        // CAUTION: 'result' is not initialized.
        // CAUTION: 'result' is only updated by the first thread.

        SHARED MatrixElement<Basis_t> matrix_element;

        SINGLE {
            matrix_element = this->pauli_strings[n].apply(configuration);
            matrix_element.coefficient *= this->coefficients[n];
        }
        SYNC;

        if(configuration != matrix_element.vector) {
            // off-diagonal string
            psi.update_input_units(configuration, matrix_element.vector, payload);

            SHARED typename Psi_t::dtype log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.vector, payload);
            SINGLE {
                result += matrix_element.coefficient * exp(log_psi_prime - log_psi);
            }

            psi.update_input_units(matrix_element.vector, configuration, payload);
        }
        else {
            // diagonal string
            SINGLE {
                result += matrix_element.coefficient;
            }
        }

        SYNC;
    }

    template<typename Psi_t, typename Basis_t>
    HDINLINE
    void local_energy(
        complex_t& result,
        const Psi_t& psi,
        const Basis_t& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload,
        const unsigned int shift = 0u,
        const bool init = true
    ) const {
        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;
        // CAUTION: 'result' is only updated by the first thread.

        SINGLE {
            if(init) {
                result = typename Psi_t::dtype(0.0);
            }
        }

        SHARED_MEM_LOOP_BEGIN(n, this->num_strings) {

            this->nth_local_energy(
                result,
                n,
                psi,
                configuration,
                log_psi,
                payload,
                shift
            );

            SHARED_MEM_LOOP_END(n);
        }
    }

    template<typename Basis_t>
    HDINLINE
    void fast_local_energy(
        complex_t& result,
        const Basis_t& configuration
    ) const {
        result = complex_t(0.0);

        for(auto n = 0u; n < this->num_strings; n++) {
            result += this->coefficients[n] * this->pauli_strings[n].apply(configuration).coefficient;
        }
    }


#endif // __CUDACC__

    inline Operator& kernel() {
        return *this;
    }

    inline const Operator& kernel() const {
        return *this;
    }

};

} // namespace kernel


struct Operator : public kernel::Operator {
    bool                gpu;
    Array<complex_t>    coefficients_ar;
    Array<PauliString>  pauli_strings_ar;

    using Kernel = kernel::Operator;

    Operator(
        const ::quantum_expression::PauliExpression& expr,
        const bool gpu
    );

    inline Operator(const Operator& other)
    :
    gpu(other.gpu),
    coefficients_ar(other.coefficients_ar),
    pauli_strings_ar(other.pauli_strings_ar)
    {
        this->kernel().num_strings = this->coefficients_ar.size();
        this->kernel().coefficients = this->coefficients_ar.data();
        this->kernel().pauli_strings = this->pauli_strings_ar.data();
    }

    ::quantum_expression::PauliExpression to_expr() const;
    vector<::quantum_expression::PauliExpression> to_expr_list() const;
};

} // namespace ann_on_gpu
