#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_Operator_cpp

#include "operator/Operator.hpp"
#include <algorithm>


namespace ann_on_gpu {

using quantum_expression::PauliExpression;
using quantum_expression::FastPauliString;


Operator::Operator(
    const PauliExpression& expr,
    const bool gpu
) : gpu(gpu), coefficients_ar(expr.size(), gpu), pauli_strings_ar(expr.size(), gpu) {

    auto i = 0u;
    for(const auto& term : expr) {
        this->coefficients_ar[i] = term.second;
        this->pauli_strings_ar[i] = PauliString(term.first.a, term.first.b);

        i++;
    }

    this->coefficients_ar.update_device();
    this->pauli_strings_ar.update_device();

    this->coefficients = this->coefficients_ar.data();
    this->pauli_strings = this->pauli_strings_ar.data();
    this->num_strings = expr.size();
}

PauliExpression Operator::to_expr() const {
    PauliExpression result;

    for(auto i = 0u; i < this->coefficients_ar.size(); i++) {
        const auto coefficient = this->coefficients_ar[i];
        const auto pauli_string = this->pauli_strings_ar[i];

        result += PauliExpression(
            FastPauliString(pauli_string.a, pauli_string.b),
            coefficient.to_std()
        );
    }

    return result;
}

vector<PauliExpression> Operator::to_expr_list() const {
    vector<PauliExpression> result;

    for(auto i = 0u; i < this->coefficients_ar.size(); i++) {
        const auto coefficient = this->coefficients_ar[i];
        const auto pauli_string = this->pauli_strings_ar[i];

        result.push_back(PauliExpression(
            FastPauliString(pauli_string.a, pauli_string.b),
            coefficient.to_std()
        ));
    }

    return result;
}


} // namespace ann_on_gpu
