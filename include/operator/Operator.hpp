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
    void local_energy(
        typename Psi_t::dtype& result,
        const Psi_t& psi,
        const Basis_t& basis_vector,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::dtype* angles,
        typename Psi_t::dtype* activations
    ) const {
        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;
        // CAUTION: 'result' is only updated by the first thread.

        SINGLE {
            result = typename Psi_t::dtype(0.0);
        }

        SHARED MatrixElement<Basis_t>   matrix_element;
        SHARED typename Psi_t::dtype    angles_prime[Psi_t::max_width];

        SHARED_MEM_LOOP_BEGIN(n, this->num_strings) {

            MULTI(j, psi.get_num_angles()) {
                angles_prime[j] = angles[j];
            }
            SINGLE {
                matrix_element = this->pauli_strings[n].apply(basis_vector);
                matrix_element.coefficient *= this->coefficients[n];
            }

            psi.update_input_units(angles_prime, basis_vector, matrix_element.vector);
            SYNC;

            SHARED typename Psi_t::dtype log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.vector, angles_prime, activations);
            SINGLE {
                result += matrix_element.coefficient * exp(log_psi_prime - log_psi);
            }

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
};

} // namespace ann_on_gpu
