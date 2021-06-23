#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_Operator_cpp

#include "basis/PauliString.hpp"
#include "basis/FermiString.hpp"
#include "operator/Operator.hpp"
#include <algorithm>


namespace ann_on_gpu {

using quantum_expression::QuantumExpression;
using quantum_expression::FastPauliString;
using quantum_expression::FermionString;

#ifdef ENABLE_SPINS

template<>
template<typename expr_t>
StandartOperator<PauliString>::StandartOperator(
    const expr_t& expr,
    const bool gpu
) : gpu(gpu), coefficients(expr.size(), gpu), quantum_strings(expr.size(), gpu) {

    auto i = 0u;
    for(const auto& term : expr) {
        this->coefficients[i] = term.second;
        this->quantum_strings[i] = PauliString(term.first.a, term.first.b);

        i++;
    }

    this->coefficients.update_device();
    this->quantum_strings.update_device();

    this->kernel().num_strings = expr.size();
    this->kernel().coefficients = this->coefficients.data();
    this->kernel().quantum_strings = this->quantum_strings.data();
}

#endif // ENABLE_SPINS


#ifdef ENABLE_FERMIONS

template<>
template<>
StandartOperator<FermiString>::StandartOperator(
    const QuantumExpression<FermionString>& expr,
    const bool gpu
) : gpu(gpu), coefficients(expr.size(), gpu), quantum_strings(expr.size(), gpu) {

    auto i = 0u;
    for(const auto& term : expr) {
        this->coefficients[i] = term.second;
        this->quantum_strings[i] = FermiString(term.first);

        i++;
    }

    this->coefficients.update_device();
    this->quantum_strings.update_device();

    this->kernel().num_strings = expr.size();
    this->kernel().coefficients = this->coefficients.data();
    this->kernel().quantum_strings = this->quantum_strings.data();
}

#endif  // ENABLE_FERMIONS

// PauliExpression Operator::to_expr() const {
//     PauliExpression result;

//     for(auto i = 0u; i < this->coefficients_ar.size(); i++) {
//         const auto coefficient = this->coefficients_ar[i];
//         const auto pauli_string = this->pauli_strings_ar[i];

//         result += PauliExpression(
//             FastPauliString(pauli_string.a, pauli_string.b),
//             coefficient.to_std()
//         );
//     }

//     return result;
// }

// vector<PauliExpression> Operator::to_expr_list() const {
//     vector<PauliExpression> result;

//     for(auto i = 0u; i < this->coefficients_ar.size(); i++) {
//         const auto coefficient = this->coefficients_ar[i];
//         const auto pauli_string = this->pauli_strings_ar[i];

//         result.push_back(PauliExpression(
//             FastPauliString(pauli_string.a, pauli_string.b),
//             coefficient.to_std()
//         ));
//     }

//     return result;
// }


} // namespace ann_on_gpu
