#ifdef ENABLE_PSI_CLASSICAL

#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/PsiFullyPolarized.hpp"

#ifdef ENABLE_PSI_CLASSICAL_ANN
#include "quantum_state/PsiDeep.hpp"
#endif // ENABLE_PSI_CLASSICAL_ANN


namespace ann_on_gpu {


template<typename dtype, unsigned int order, typename PsiRef>
void PsiClassical_t<dtype, order, PsiRef>::init_kernel() {
    this->kernel().params = this->params.data();
    this->kernel().H_local_diagonal = this->H_local_diagonal_op.kernel();
    this->kernel().H_local = this->H_local_op.kernel();
    this->kernel().H_2_local = this->H_2_local_op.kernel();

    this->num_params = this->params.size();
    this->num_local_energies = this->num_sites * this->H_local.num_strings;

    if(order > 1u) {
        this->num_local_energies += this->H_2_local.num_strings;

        this->m_2.begin_local_energies = this->num_sites * this->H_local.num_strings;
        this->m_2.begin_params = this->H_local_diagonal.num_strings + this->H_local.num_strings;
        this->m_2.end_params = this->m_2.begin_params + this->H_2_local.num_strings;

        // std::cout << "this->m_2.begin_local_energies: " << this->m_2.begin_local_energies << std::endl;
        // std::cout << "this->m_2.begin_params: " << this->m_2.begin_params << std::endl;
        // std::cout << "this->m_2.end_params: " << this->m_2.end_params << std::endl;

        this->m_1_squared.begin_params = this->m_2.end_params;

        this->m_1_squared.num_ll_pairs = this->H_local.num_strings * (this->H_local.num_strings + 1u) / 2u;

        for(auto l = 0u; l < this->H_local.num_strings; l++) {
            for(auto l_prime = 0u; l_prime <= l; l_prime++) {
                this->ids_l.push_back(l);
                this->ids_l_prime.push_back(l_prime);
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


template struct PsiClassical_t<complex_t, 1u, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, 2u, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, 1u, PsiDeep>;
template struct PsiClassical_t<complex_t, 2u, PsiDeep>;
#endif // ENABLE_PSI_CLASSICAL_ANN


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CLASSICAL
