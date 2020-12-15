#ifdef ENABLE_PSI_CLASSICAL

#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/PsiFullyPolarized.hpp"

#ifdef ENABLE_PSI_CLASSICAL_ANN
#include "quantum_state/PsiDeep.hpp"
#endif // ENABLE_PSI_CLASSICAL_ANN


namespace ann_on_gpu {


template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
void PsiClassical_t<dtype, Operator_t, order, symmetric, PsiRef>::init_kernel() {
    this->kernel().params = this->params.data();

    this->kernel().num_ops_H = this->H_local.size();
    this->kernel().num_ops_H_2 = this->H_2_local.size();

    for(auto i = 0u; i < this->H_local.size(); i++) {
        this->H_local_kernel[i] = this->H_local[i].kernel();
    }
    for(auto i = 0u; i < this->H_2_local.size(); i++) {
        this->H_2_local_kernel[i] = this->H_2_local[i].kernel();
    }

    this->H_local_kernel.update_device();
    this->H_2_local_kernel.update_device();

    this->kernel().H_local = this->H_local_kernel.data();
    this->kernel().H_2_local = this->H_2_local_kernel.data();

    this->num_params = this->params.size();

    if(order > 1u) {
        if(symmetric) {
            this->m_2.begin_local_energies = this->num_sites * this->H_local.size();
        }
        else {
            this->m_2.begin_local_energies = this->H_local.size();
        }

        this->m_2.begin_params = this->H_local.size();
        this->m_2.end_params = this->m_2.begin_params + this->H_2_local.size();

        // std::cout << "this->m_2.begin_local_energies: " << this->m_2.begin_local_energies << std::endl;
        // std::cout << "this->m_2.begin_params: " << this->m_2.begin_params << std::endl;
        // std::cout << "this->m_2.end_params: " << this->m_2.end_params << std::endl;

        this->m_1_squared.begin_params = this->m_2.end_params;

        if(symmetric) {
            this->m_1_squared.num_ll_pairs = this->H_local.size() * (this->H_local.size() + 1u) / 2u;

            for(auto l = 0u; l < this->H_local.size(); l++) {
                for(auto l_prime = 0u; l_prime <= l; l_prime++) {
                    this->ids_l.push_back(l);
                    this->ids_l_prime.push_back(l_prime);
                }
            }
        }
        else {
            const auto cell_size = this->H_local.size() / this->num_sites;
            const auto num_pairs = this->num_params - this->m_1_squared.begin_params;
            const auto distance = num_pairs / (this->H_local.size() * cell_size);

            this->m_1_squared.num_ll_pairs = num_pairs;

            for(auto i = 0u; i < this->H_local.size(); i++) {
                const auto cell_i = i / cell_size;

                for(auto cell_j = cell_i; cell_j < cell_i + distance; cell_j++) {
                    const auto cell_j_offset = (cell_j % distance) * cell_size;

                    for(auto j = cell_j_offset; j < cell_j_offset + cell_size; j++) {
                        this->ids_l.push_back(i);
                        this->ids_l_prime.push_back(j);
                    }
                }
            }
        }

        this->ids_l.resize(this->m_1_squared.num_ll_pairs);
        this->ids_l_prime.resize(this->m_1_squared.num_ll_pairs);

        this->ids_l.update_device();
        this->ids_l_prime.update_device();

        this->kernel().m_1_squared.ids_l = this->ids_l.data();
        this->kernel().m_1_squared.ids_l_prime = this->ids_l_prime.data();
    }

    this->kernel().psi_ref = this->psi_ref.kernel();
}

#ifdef PSI_CLASSICAL_SYMMETRIC

template struct PsiClassical_t<complex_t, Operator_t, 1u, true, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, true, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, Operator_t, 1u, true, PsiDeep>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, true, PsiDeep>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#else

template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiDeep>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiDeep>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#endif // PSI_CLASSICAL_SYMMETRIC


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CLASSICAL
