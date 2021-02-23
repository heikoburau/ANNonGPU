#pragma once

#include "operator/SparseMatrix.hpp"
#include "bases.hpp"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#endif // __PYTHONCC__


namespace ann_on_gpu {

namespace kernel {

struct SuperOperator {

    SparseMatrix*       matrices;
    unsigned int        num_matrices;

#ifdef __CUDACC__

#ifdef ENABLE_SPINS
    template<typename Psi_t>
    HDINLINE
    void nth_local_energy(
        typename Psi_t::dtype& result,
        unsigned int n,
        const Psi_t& psi,
        const Spins& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload,
        const unsigned int shift = 0u
    ) const {
        // not supported
    }

#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS

    template<typename Psi_t>
    HDINLINE
    void nth_local_energy(
        complex_t& result,
        unsigned int n,
        const Psi_t& psi,
        const PauliString& configuration,
        const typename Psi_t::dtype& log_psi,
        typename Psi_t::Payload& payload,
        const unsigned int shift = 0u
    ) const {
        #include "cuda_kernel_defines.h"
        using dtype = typename Psi_t::dtype;
        // CAUTION: 'result' is not initialized.
        // CAUTION: 'result' is only updated by the first thread.

        SHARED SparseMatrix                matrix;
        SHARED MatrixElement<PauliString>  matrix_element;

        SINGLE {
            matrix_element.vector = configuration;

            matrix = this->matrices[n];
            if(shift) {
                matrix.site_i = (matrix.site_i + shift) % psi.num_sites;
                matrix.site_j = (matrix.site_j + shift) % psi.num_sites;
            }
            this->matrices[n].apply(matrix_element);
        }
        SYNC;

        if(matrix_element.coefficient != complex_t(0.0) && configuration != matrix_element.vector) {
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

#endif // ENABLE_PAULIS

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
                result = complex_t(0.0);
            }
        }

        SHARED_MEM_LOOP_BEGIN(n, this->num_matrices) {

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


#endif // __CUDACC__

    inline SuperOperator& kernel() {
        return *this;
    }

    inline const SuperOperator& kernel() const {
        return *this;
    }
};


} // namespace kernel



struct SuperOperator : public kernel::SuperOperator {
    using Kernel = kernel::SuperOperator;

    bool                    gpu;

    Array<SparseMatrix>     matrices;

    SuperOperator() = delete;
    inline SuperOperator(const SuperOperator& other)
    :
    gpu(other.gpu),
    matrices(other.matrices)
    {
        this->kernel().matrices = this->matrices.data();
        this->kernel().num_matrices = this->matrices.size();
    }

#ifdef __PYTHONCC__

    inline SuperOperator(
        const vector<SparseMatrix>& matrices,
        const bool gpu
    ) : gpu(gpu), matrices(matrices.size(), gpu) {

        copy(matrices.begin(), matrices.end(), this->matrices.begin());

        this->matrices.update_device();

        this->kernel().matrices = this->matrices.data();
        this->kernel().num_matrices = matrices.size();
    }

#endif // __PYTHONCC__

};


} // namespace ann_on_gpu
