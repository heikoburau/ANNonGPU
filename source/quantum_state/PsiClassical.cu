#ifdef ENABLE_PSI_CLASSICAL

#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/PsiFullyPolarized.hpp"

namespace ann_on_gpu {


template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
void PsiClassical_t<dtype, Operator_t, order, symmetric, PsiRef>::init_kernel() {
    this->kernel().params = this->params.data();

    this->kernel().num_ops_H = this->H_local.size();

    for(auto i = 0u; i < this->H_local.size(); i++) {
        this->H_local_kernel[i] = this->H_local[i].kernel();
    }

    this->H_local_kernel.update_device();

    this->kernel().H_local = this->H_local_kernel.data();

    this->num_params = this->params.size();

    if(order > 1u) {
        this->m_1_squared.begin_params = this->H_local.size();

        this->m_1_squared.num_ll_pairs = this->ids_l.size();

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
template struct PsiClassical_t<complex_t, Operator_t, 1u, true, PsiExact>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, true, PsiExact>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#else

template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiExact>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiExact>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#endif // PSI_CLASSICAL_SYMMETRIC


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CLASSICAL
