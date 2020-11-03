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
    this->num_params = this->params.size();

    this->kernel().H_local = this->H_local_op.kernel();

    if(order > 1u) {
        this->kernel().m_2.symmetric_terms = this->M_2_op.kernel();
        this->kernel().m_2.begin = this->H_local_op.num_strings;
        this->kernel().m_2.end = this->kernel().m_2.begin + this->M_2_op.num_strings;

        this->kernel().m_1_squared.terms = this->M_1_squared_op.kernel();
        this->kernel().m_1_squared.begin = this->kernel().m_2.end;
        const auto N = this->M_1_squared_op.num_strings;
        this->kernel().m_1_squared.num_pairs = N;
        this->kernel().m_1_squared.end = this->kernel().m_1_squared.begin + N * (N + 1u) / 2u;

        for(auto i = 0u; i < N; i++) {
            for(auto j = 0u; j <= i; j++) {
                this->ids_i.push_back(i);
                this->ids_j.push_back(j);
            }
        }

        this->ids_i.resize(N);
        this->ids_j.resize(N);

        this->ids_i.update_device();
        this->ids_j.update_device();

        this->kernel().m_1_squared.ids_i = this->ids_i.data();
        this->kernel().m_1_squared.ids_j = this->ids_j.data();
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
