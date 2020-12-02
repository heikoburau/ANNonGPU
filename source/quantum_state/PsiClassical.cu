#ifdef ENABLE_PSI_CLASSICAL

#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/PsiFullyPolarized.hpp"

#ifdef ENABLE_PSI_CLASSICAL_ANN
#include "quantum_state/PsiDeep.hpp"
#endif // ENABLE_PSI_CLASSICAL_ANN


namespace ann_on_gpu {


template<typename dtype, typename Operator_t, unsigned int order, typename PsiRef>
void PsiClassical_t<dtype, Operator_t, order, PsiRef>::init_kernel() {
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
        this->m_2.begin_local_energies = this->num_sites * this->H_local.size();
        this->m_2.begin_params = this->H_local.size();
        this->m_2.end_params = this->m_2.begin_params + this->H_2_local.size();

        // std::cout << "this->m_2.begin_local_energies: " << this->m_2.begin_local_energies << std::endl;
        // std::cout << "this->m_2.begin_params: " << this->m_2.begin_params << std::endl;
        // std::cout << "this->m_2.end_params: " << this->m_2.end_params << std::endl;

        this->m_1_squared.begin_params = this->m_2.end_params;

        this->m_1_squared.num_ll_pairs = this->H_local.size() * (this->H_local.size() + 1u) / 2u;

        for(auto l = 0u; l < this->H_local.size(); l++) {
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


template struct PsiClassical_t<complex_t, SuperOperator, 1u, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, SuperOperator, 2u, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, SuperOperator, 1u, PsiDeep>;
template struct PsiClassical_t<complex_t, SuperOperator, 2u, PsiDeep>;
#endif // ENABLE_PSI_CLASSICAL_ANN


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CLASSICAL
