#ifdef ENABLE_PSI_CLASSICAL

#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/PsiFullyPolarized.hpp"

#include <algorithm>

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
        this->num_params += this->psi_ref.num_params;
    }

    this->kernel().psi_ref = this->psi_ref.kernel();
}

template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
Array<dtype> PsiClassical_t<dtype, Operator_t, order, symmetric, PsiRef>::get_params() const {
    Array<dtype> result(this->num_params, false);

    copy(this->params.begin(), this->params.end(), result.begin());

    if(order > 1u) {
        const auto psi_ref_params = this->psi_ref.get_params();
        copy(psi_ref_params.begin(), psi_ref_params.end(), result.begin() + this->params.size());
    }

    return result;
}

template<typename dtype, typename Operator_t, unsigned int order, bool symmetric, typename PsiRef>
void PsiClassical_t<dtype, Operator_t, order, symmetric, PsiRef>::set_params(const Array<dtype> & new_params) {
    copy(
        new_params.begin(),
        new_params.begin() + this->params.size(),
        this->params.begin()
    );
    this->params.update_device();

    if(order > 1u) {
        Array<dtype> new_params_ref(this->psi_ref.num_params, false);

        copy(
            new_params.begin() + this->params.size(),
            new_params.end(),
            new_params_ref.begin()
        );

        this->psi_ref.set_params(new_params_ref);

        this->update_psi_ref_kernel();
    }
}

#ifdef PSI_CLASSICAL_SYMMETRIC

template struct PsiClassical_t<complex_t, Operator_t, 1u, true, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, true, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, Operator_t, 1u, true, PsiRBM>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, true, PsiRBM>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#else

template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiFullyPolarized>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiFullyPolarized>;

#ifdef ENABLE_PSI_CLASSICAL_ANN
template struct PsiClassical_t<complex_t, Operator_t, 1u, false, PsiRBM>;
template struct PsiClassical_t<complex_t, Operator_t, 2u, false, PsiRBM>;
#endif // ENABLE_PSI_CLASSICAL_ANN

#endif // PSI_CLASSICAL_SYMMETRIC


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CLASSICAL
