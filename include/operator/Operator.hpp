#pragma once

#include "operator/MatrixElement.hpp"
#include "bases.hpp"
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

template<typename QuantumString_t>
struct StandartOperator {
    using QuantumString = QuantumString_t;

    complex_t*        coefficients;
    QuantumString*    quantum_strings;
    unsigned int      num_strings;

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
            matrix_element = this->quantum_strings[n].apply(configuration);
            matrix_element.coefficient *= this->coefficients[n];
        }
        SYNC;

        if(configuration != matrix_element.vector) {
            // off-diagonal string
            psi.update_input_units(configuration, matrix_element.vector, payload);

            SHARED typename Psi_t::dtype log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.vector, payload);
            SINGLE {
                auto diff = log_psi_prime - log_psi;

                diff.__re_ = min(diff.__re_, 10.0);

                result += matrix_element.coefficient * exp(diff);
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
    complex_t fast_local_energy(
        const Basis_t& configuration
    ) const {
        complex_t result(0.0);

        for(auto n = 0u; n < this->num_strings; n++) {
            result += this->coefficients[n] * this->quantum_strings[n].apply(configuration).coefficient;
        }

        return result;
    }

    template<typename Basis_t>
    HDINLINE
    void fast_local_energy_parallel(
        complex_t& result,
        const Basis_t& configuration
    ) const {
        SINGLE {
            result = complex_t(0.0);
        }
        SYNC;

        LOOP(n, this->num_strings) {
            generic_atomicAdd(
                &result,
                this->coefficients[n] * this->quantum_strings[n].apply(configuration).coefficient
            );
        }
        SYNC;
    }


#endif // __CUDACC__

    inline StandartOperator& kernel() {
        return *this;
    }

    inline const StandartOperator& kernel() const {
        return *this;
    }

};

} // namespace kernel


template<typename QuantumString_t>
struct StandartOperator : public kernel::StandartOperator<QuantumString_t> {
    using QuantumString = QuantumString_t;

    bool                  gpu;
    Array<complex_t>      coefficients;
    Array<QuantumString>  quantum_strings;

    using Kernel = kernel::StandartOperator<QuantumString_t>;

    template<typename expr_t>
    StandartOperator(
        const expr_t& expr,
        const bool gpu
    );

    inline StandartOperator(const StandartOperator& other)
    :
    gpu(other.gpu),
    coefficients(other.coefficients),
    quantum_strings(other.quantum_strings)
    {
        this->kernel().num_strings = this->coefficients.size();
        this->kernel().coefficients = this->coefficients.data();
        this->kernel().quantum_strings = this->quantum_strings.data();
    }

    // ::quantum_expression::PauliExpression to_expr() const;
    // vector<::quantum_expression::PauliExpression> to_expr_list() const;
};


#ifdef ENABLE_SPINS
using Operator = StandartOperator<PauliString>;
#endif

#ifdef ENABLE_FERMIONS
using Operator = StandartOperator<Fermions>;
#endif


} // namespace ann_on_gpu
