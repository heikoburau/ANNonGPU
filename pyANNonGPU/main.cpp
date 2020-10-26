#define __PYTHONCC__
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/KullbackLeibler.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"
#include "network_functions/PsiAngles.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"
#include "operator/Operator.hpp"
#include "bases.hpp"
#include "RNGStates.hpp"
#include "types.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor/xadapt.hpp"
#include <xtensor-python/pytensor.hpp>

#include <iostream>
#include <complex>


namespace py = pybind11;

using namespace ann_on_gpu;
using namespace pybind11::literals;

template<unsigned int dim, typename real_dtype = double>
using complex_tensor = xt::pytensor<std::complex<real_dtype>, dim>;

template<unsigned int dim, typename dtype = double>
using real_tensor = xt::pytensor<dtype, dim>;

// Python Module and Docstrings

PYBIND11_MODULE(_pyANNonGPU, m)
{
    xt::import_numpy();

#ifdef ENABLE_PSI_DEEP
    py::class_<PsiDeep>(m, "PsiDeep")
        .def(py::init<
            const complex_tensor<1u, PsiDeep::real_dtype>&,
            const vector<complex_tensor<1u, PsiDeep::real_dtype>>&,
            const vector<xt::pytensor<unsigned int, 2u>>&,
            const vector<complex_tensor<2u, PsiDeep::real_dtype>>&,
            const complex_tensor<1u, PsiDeep::real_dtype>&,
            const double,
            const bool,
            const bool
        >())
        .def("copy", &PsiDeep::copy)
        .def_readwrite("prefactor", &PsiDeep::prefactor)
        .def_readwrite("translational_invariance", &PsiDeep::translational_invariance)
        .def_readwrite("N_i", &PsiDeep::N_i)
        .def_readwrite("N_j", &PsiDeep::N_j)
        .def_readonly("gpu", &PsiDeep::gpu)
        .def_readonly("N", &PsiDeep::N)
        .def_readonly("num_params", &PsiDeep::num_params)
        .def_property(
            "params",
            [](const PsiDeep& psi) {return psi.get_params().to_pytensor_1d();},
            [](PsiDeep& psi, const complex_tensor<1u, PsiDeep::real_dtype>& new_params) {
                psi.set_params(Array<typename PsiDeep::dtype>(new_params, false));
            }
        )
        .def_property_readonly("a", [](const PsiDeep& psi) {return psi.input_biases.to_pytensor_1d();})
        .def_property_readonly("b", &PsiDeep::get_b)
        .def_property_readonly("connections", &PsiDeep::get_connections)
        .def_property_readonly("W", &PsiDeep::get_W)
        .def_property_readonly("input_biases", [](const PsiDeep& psi) {return psi.input_biases.to_pytensor_1d();})
        .def_property_readonly("final_weights", [](const PsiDeep& psi) {return psi.final_weights.to_pytensor_1d();})
    #ifdef ENABLE_EXACT_SUMMATION
    #ifdef ENABLE_SPINS
        .def("_vector", [](const PsiDeep& psi, ExactSummationSpins& exact_summation) {return psi_vector_py(psi, exact_summation);})
        .def("norm", [](const PsiDeep& psi, ExactSummationSpins& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_SPINS
    #ifdef ENABLE_PAULIS
        .def("_vector", [](const PsiDeep& psi, ExactSummationPaulis& exact_summation) {return psi_vector_py(psi, exact_summation);})
        .def("norm", [](const PsiDeep& psi, ExactSummationPaulis& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_PAULIS
    #endif // ENABLE_EXACT_SUMMATION
        ;

#endif // ENABLE_PSI_DEEP

    py::class_<Operator>(m, "Operator")
        .def(py::init<
            const quantum_expression::PauliExpression&,
            const bool
        >())
        .def_property_readonly("expr", &Operator::to_expr)
        .def_readonly("num_strings", &Operator::num_strings);

#ifdef ENABLE_SPINS
    py::class_<ann_on_gpu::Spins>(m, "Spins")
        .def(py::init<ann_on_gpu::Spins::dtype, const unsigned int>())
        .def("array", &ann_on_gpu::Spins::array);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
    py::class_<ann_on_gpu::PauliString>(m, "PauliString")
        .def(py::init<ann_on_gpu::PauliString::dtype, ann_on_gpu::PauliString::dtype>())
        .def("array", &ann_on_gpu::PauliString::array);
#endif // ENABLE_PAULIS


#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_SPINS
    py::class_<MonteCarloSpins>(m, "MonteCarloSpins")
        .def(py::init(&make_MonteCarloSpins))
        .def(py::init<MonteCarloSpins&>())
        // .def("set_total_z_symmetry", &MonteCarloSpins::set_total_z_symmetry)
        .def_property_readonly("num_steps", &MonteCarloSpins::get_num_steps)
        .def_property_readonly("acceptance_rate", [](const MonteCarloSpins& mc){
            return float(mc.acceptances_ar.front()) / float(mc.acceptances_ar.front() + mc.rejections_ar.front());
        });
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
    py::class_<MonteCarloPaulis>(m, "MonteCarloPaulis")
        .def(py::init(&make_MonteCarloPaulis))
        .def(py::init<MonteCarloPaulis&>())
        // .def("set_total_z_symmetry", &MonteCarloPaulis::set_total_z_symmetry)
        .def_property_readonly("num_steps", &MonteCarloPaulis::get_num_steps)
        .def_property_readonly("acceptance_rate", [](const MonteCarloPaulis& mc){
            return float(mc.acceptances_ar.front()) / float(mc.acceptances_ar.front() + mc.rejections_ar.front());
        });
#endif // ENABLE_PAULIS
#endif // ENABLE_MONTE_CARLO


#ifdef ENABLE_EXACT_SUMMATION
#ifdef ENABLE_SPINS
    py::class_<ExactSummationSpins>(m, "ExactSummationSpins")
        .def(py::init<unsigned int, bool>())
        // .def("set_total_z_symmetry", &ExactSummationSpins::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummationSpins::get_num_steps);
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
    py::class_<ExactSummationPaulis>(m, "ExactSummationPaulis")
        .def(py::init<unsigned int, bool>())
        // .def("set_total_z_symmetry", &ExactSummationPaulis::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummationPaulis::get_num_steps);
#endif // ENABLE_PAULIS
#endif // ENABLE_EXACT_SUMMATION


    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_SPINS
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarloSpins>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarloSpins>)
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarloPaulis>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarloPaulis>)
#endif // ENABLE_PAULIS
#endif // ENABLE_MONTE_CARLO
#ifdef ENABLE_EXACT_SUMMATION
#ifdef ENABLE_SPINS
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummationSpins>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummationSpins>)
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummationPaulis>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummationPaulis>)
#endif // ENABLE_PAULIS
#endif // ENABLE_EXACT_SUMMATION
    ;


    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, unsigned int, bool>())
#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_SPINS
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarloSpins>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarloSpins>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarloPaulis>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarloPaulis>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PAULIS
#endif // ENABLE_MONTE_CARLO
#ifdef ENABLE_MONTE_EXACT_SUMMATION
#ifdef ENABLE_SPINS
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummationSpins>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummationSpins>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummationPaulis>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummationPaulis>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PAULIS
#endif // ENABLE_MONTE_EXACT_SUMMATION
    ;


//     py::class_<KullbackLeibler>(m, "KullbackLeibler")
//         .def(py::init<unsigned int, bool>())
// #ifdef ENABLE_MONTE_CARLO
// #ifdef ENABLE_PSI_DEEP
//         .def("__call__", &KullbackLeibler::value_with_op<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
//         .def("gradient", &KullbackLeibler::gradient_with_op_py<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
//         .def("__call__", &KullbackLeibler::value_2nd_order<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "op"_a, "op2"_a, "spin_ensemble"_a)
//         .def("gradient", &KullbackLeibler::gradient_2nd_order_py<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "op"_a, "op2"_a, "spin_ensemble"_a, "nu"_a)
// #endif // ENABLE_PSI_DEEP
// #endif // ENABLE_MONTE_CARLO
// #ifdef ENABLE_EXACT_SUMMATION
// #ifdef ENABLE_PSI_DEEP
//         .def("__call__", &KullbackLeibler::value_with_op<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
//         .def("gradient", &KullbackLeibler::gradient_with_op_py<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
//         .def("__call__", &KullbackLeibler::value_2nd_order<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "op"_a, "op2"_a, "spin_ensemble"_a)
//         .def("gradient", &KullbackLeibler::gradient_2nd_order_py<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "op"_a, "op2"_a, "spin_ensemble"_a, "nu"_a)
// #endif // ENABLE_PSI_DEEP
// #endif // ENABLE_EXACT_SUMMATION
//     ;


    py::class_<RNGStates>(m, "RNGStates")
        .def(py::init<unsigned int, bool>());


    m.def("activation_function", [](const complex<double>& x) {
        return my_logcosh(complex_t(x.real(), x.imag())).to_std();
    });

    m.def("setDevice", setDevice);
    m.def("start_profiling", start_profiling);
    m.def("stop_profiling", stop_profiling);
}
