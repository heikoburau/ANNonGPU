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
        typename Psi_t::dtype& result,
        unsigned int n,
        const Psi_t& psi,
        const Basis_t& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload
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
    void nth_local_energy_symmetric(
        typename Psi_t::dtype& result,
        const unsigned int n,
        const Psi_t& psi,
        const Basis_t& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload
    ) const {
        // 'symmetric' means that the result will be invariant under translation of the given configuration.
        // A symmetric psi is implied, therefore only a single call to psi(s) is performed if any.

        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;

        if(this->pauli_strings[n].applies_a_prefactor(configuration)) {
            SINGLE {
                result = typename Psi_t::dtype(0.0);
            }
            SYNC;
            MULTI(i, psi.num_sites) {
                generic_atomicAdd(
                    &result,
                    this->coefficients[n] * this->pauli_strings[n].apply(
                        configuration.rotate_left(i, psi.num_sites)
                    ).coefficient * (1.0 / psi.num_sites)
                );
            }
        }
        else {
            SINGLE {
                result = this->coefficients[n];
            }
        }

        if(!this->pauli_strings[n].is_diagonal_on_basis(configuration)) {
            SHARED Basis_t configuration_prime;

            SINGLE {
                configuration_prime = this->pauli_strings[n].apply(configuration).vector;
            }
            SYNC;

            psi.update_input_units(configuration, configuration_prime, payload);

            SHARED typename Psi_t::dtype log_psi_prime;
            psi.log_psi_s(log_psi_prime, configuration_prime, payload);
            SINGLE {
                result *= exp(log_psi_prime - log_psi);
            }

            psi.update_input_units(configuration_prime, configuration, payload);
        }

        // printf("conf: %lu\n", configuration.configuration());
        // printf("h: %lu, %lu\n", this->pauli_strings[n].a, this->pauli_strings[n].b);
        // printf("E_loc: %f, %f\n", result.real(), result.imag());

        SYNC;
    }

    template<typename Psi_t, typename Basis_t>
    HDINLINE
    void local_energy(
        typename Psi_t::dtype& result,
        const Psi_t& psi,
        const Basis_t& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload
    ) const {
        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;
        // CAUTION: 'result' is only updated by the first thread.

        SINGLE {
            result = typename Psi_t::dtype(0.0);
        }

        SHARED_MEM_LOOP_BEGIN(n, this->num_strings) {

            this->nth_local_energy(
                result,
                n,
                psi,
                configuration,
                log_psi,
                payload
            );

            SHARED_MEM_LOOP_END(n);
        }
    }


#endif // __CUDACC__

    inline Operator kernel() const {
        return *this;
    }
};

} // namespace kernel


struct Operator : public kernel::Operator {
    bool                gpu;
    Array<complex_t>    coefficients_ar;
    Array<PauliString>  pauli_strings_ar;

    Operator(
        const ::quantum_expression::PauliExpression& expr,
        const bool gpu
    );

    ::quantum_expression::PauliExpression to_expr() const;
    vector<::quantum_expression::PauliExpression> to_expr_list() const;
};

} // namespace ann_on_gpu
