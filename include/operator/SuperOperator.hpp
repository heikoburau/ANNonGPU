#pragma once

#include "operator/Matrix4x4.hpp"
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

    static constexpr auto max_string_length = 8u;

    complex_t*      coefficients;
    Matrix4x4*      matrices;
    unsigned int*   string_lengths;
    unsigned int    num_strings;

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
        typename Psi_t::dtype& result,
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

        SHARED Matrix4x4                   matrix;
        SHARED MatrixElement<PauliString>  matrix_element;

        SINGLE {
            matrix_element.coefficient = this->coefficients[n];
            matrix_element.vector = configuration;

            for(auto m = 0u; m < this->string_lengths[n]; m++) {
                matrix = this->matrices[n * max_string_length + m];
                if(shift) {
                    matrix.site = psi.num_sites - 1 - matrix.site;
                }

                matrix.apply(matrix_element);
            }
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

#endif // ENABLE_PAULIS

    template<typename Psi_t, typename Basis_t>
    HDINLINE
    void local_energy(
        typename Psi_t::dtype& result,
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

    Array<complex_t>        coefficients;
    Array<Matrix4x4>        matrices;
    Array<unsigned int>     string_lengths;

    SuperOperator() = delete;
    inline SuperOperator(const SuperOperator& other)
    :
    gpu(other.gpu),
    coefficients(other.coefficients),
    matrices(other.matrices),
    string_lengths(other.string_lengths)
    {
        this->kernel().coefficients = this->coefficients.data();
        this->kernel().matrices = this->matrices.data();
        this->kernel().string_lengths = this->string_lengths.data();
        this->kernel().num_strings = this->coefficients.size();
    }

#ifdef __PYTHONCC__

    inline SuperOperator(
        const vector<complex<double>>& coefficients_arg,
        const vector<vector<unsigned int>> site_indices,
        const vector<vector<xt::pytensor<complex<double>, 2u>>>& raw_matrices,
        const bool gpu
    ) : gpu(gpu), coefficients(gpu), matrices(gpu), string_lengths(gpu) {

        this->kernel().num_strings = coefficients_arg.size();

        this->coefficients.resize(this->num_strings);
        this->string_lengths.resize(this->num_strings);
        this->matrices.resize(this->num_strings * max_string_length);
        this->matrices.clear();

        for(auto n = 0u; n < this->num_strings; n++) {
            this->coefficients[n] = coefficients_arg[n];

            const auto site_indices_row = site_indices[n];
            const auto raw_matrices_row = raw_matrices[n];

            this->string_lengths[n] = site_indices_row.size();

            for(auto m = 0u; m < this->string_lengths[n]; m++) {
                auto& matrix = this->matrices[n * max_string_length + m];
                const auto raw_matrix = raw_matrices_row[m];

                matrix.site = site_indices_row[m];

                auto is_diagonal = true;
                for(auto i = 0u; i < 4u; i++) {
                    for(auto j = 0u; j < 4u; j++) {
                        if(abs(raw_matrix(i, j)) > 1e-4) {
                            matrix.values[i] = raw_matrix(i, j);
                            matrix.pauli_types[i] = j;
                            if(i != j) {
                                is_diagonal = false;
                            }
                            break;
                        }
                    }
                }
                matrix.is_diagonal = is_diagonal;
            }
        }

        this->coefficients.update_device();
        this->string_lengths.update_device();
        this->matrices.update_device();
        this->kernel().coefficients = this->coefficients.data();
        this->kernel().string_lengths = this->string_lengths.data();
        this->kernel().matrices = this->matrices.data();
    }

#endif // __PYTHONCC__

};


} // namespace ann_on_gpu
