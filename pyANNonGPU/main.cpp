// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// main.cpp.template
// ***********************************************************

#define __PYTHONCC__

#ifndef LEAN_AND_MEAN
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/KullbackLeibler.hpp"
#include "network_functions/TDVP.hpp"
#endif // LEAN_AND_MEAN

#include "network_functions/CalibratePsi.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"
#include "network_functions/PsiAngles.hpp"
#include "network_functions/ApplyOperator.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"
#include "operators.hpp"
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

    py::class_<PsiDeep>(m, "PsiDeep")
        .def(py::init<
            const unsigned int,
            const complex_tensor<1u, PsiDeep::real_dtype>&,
            const vector<complex_tensor<1u, PsiDeep::real_dtype>>&,
            const vector<xt::pytensor<unsigned int, 2u>>&,
            const vector<complex_tensor<2u, PsiDeep::real_dtype>>&,
            const complex_tensor<1u, PsiDeep::real_dtype>&,
            const double,
            const bool
        >())
        .def("copy", &PsiDeep::copy)
        .def_readwrite("num_sites", &PsiDeep::num_sites)
        .def_readwrite("prefactor", &PsiDeep::prefactor)
        .def_property(
            "log_prefactor",
            [](const PsiDeep& psi) {return psi.log_prefactor.to_std();},
            [](PsiDeep& psi, complex<double> value) {psi.log_prefactor = complex_t(value);}
        )
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
        .def_property_readonly("symmetric", [](const PsiDeep& psi){return psi.is_symmetric();})
        .def_property_readonly("a", [](const PsiDeep& psi) {return psi.input_weights.to_pytensor_1d();})
        .def_property_readonly("b", &PsiDeep::get_b)
        .def_property_readonly("connections", &PsiDeep::get_connections)
        .def_property_readonly("W", &PsiDeep::get_W)
        .def_property_readonly("input_weights", [](const PsiDeep& psi) {return psi.input_weights.to_pytensor_1d();})
        .def_property_readonly("final_weights", [](const PsiDeep& psi) {return psi.final_weights.to_pytensor_1d();})
        #ifdef ENABLE_EXACT_SUMMATION
#if defined(ENABLE_SPINS)
        .def("_vector", [](PsiDeep& psi, ExactSummation_t<Spins>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiDeep& psi, ExactSummation_t<Spins>& exact_summation) {return psi_norm(psi, exact_summation);})
#endif
#if defined(ENABLE_PAULIS)
        .def("_vector", [](PsiDeep& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiDeep& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_norm(psi, exact_summation);})
#endif
        #endif // ENABLE_EXACT_SUMMATION
        ;

#if defined(ENABLE_PSI_CLASSICAL)
    py::class_<PsiClassicalFP<1u>>(m, "PsiClassicalFP_1")
        .def(py::init<
            const unsigned int,
            const vector<Operator_t>&,
            const vector<Operator_t>&,
            const complex_tensor<1u>&,
            const typename PsiClassicalFP<1u>::PsiRef&,
            const double,
            const bool
        >())
        .def("copy", &PsiClassicalFP<1u>::copy)
        .def("__pos__", &PsiClassicalFP<1u>::copy)
        .def_readwrite("num_sites", &PsiClassicalFP<1u>::num_sites)
        .def_readonly("H_local", &PsiClassicalFP<1u>::H_local)
        .def_readonly("H_2_local", &PsiClassicalFP<1u>::H_2_local)
        .def_readwrite("prefactor", &PsiClassicalFP<1u>::prefactor)
        .def_property(
            "log_prefactor",
            [](const PsiClassicalFP<1u>& psi) {return psi.log_prefactor.to_std();},
            [](PsiClassicalFP<1u>& psi, complex<double> value) {psi.log_prefactor = complex_t(value);}
        )
        // return just a copy obviously, so writing has no effect..
        .def_property_readonly(
            "psi_ref",
            [](const PsiClassicalFP<1u>& psi) {return psi.psi_ref;}
        )
        .def_readonly("gpu", &PsiClassicalFP<1u>::gpu)
        .def_readonly("num_params", &PsiClassicalFP<1u>::num_params)
        .def_property_readonly("order", [](const PsiClassicalFP<1u>& psi) {return psi.get_order();})
        .def_property(
            "params",
            [](const PsiClassicalFP<1u>& psi) {return psi.params.to_pytensor_1d();},
            [](PsiClassicalFP<1u>& psi, const complex_tensor<1u>& new_params) {psi.params = new_params;}
        )
        .def("update_psi_ref_kernel", &PsiClassicalFP<1u>::update_psi_ref_kernel)
        #ifdef ENABLE_EXACT_SUMMATION
        #ifdef ENABLE_SPINS
        .def("vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_SPINS
        #ifdef ENABLE_PAULIS
        .def("vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_PAULIS
        #endif // ENABLE_EXACT_SUMMATION
        ;
#endif
#if defined(ENABLE_PSI_CLASSICAL)
    py::class_<PsiClassicalFP<2u>>(m, "PsiClassicalFP_2")
        .def(py::init<
            const unsigned int,
            const vector<Operator_t>&,
            const vector<Operator_t>&,
            const complex_tensor<1u>&,
            const typename PsiClassicalFP<2u>::PsiRef&,
            const double,
            const bool
        >())
        .def("copy", &PsiClassicalFP<2u>::copy)
        .def("__pos__", &PsiClassicalFP<2u>::copy)
        .def_readwrite("num_sites", &PsiClassicalFP<2u>::num_sites)
        .def_readonly("H_local", &PsiClassicalFP<2u>::H_local)
        .def_readonly("H_2_local", &PsiClassicalFP<2u>::H_2_local)
        .def_readwrite("prefactor", &PsiClassicalFP<2u>::prefactor)
        .def_property(
            "log_prefactor",
            [](const PsiClassicalFP<2u>& psi) {return psi.log_prefactor.to_std();},
            [](PsiClassicalFP<2u>& psi, complex<double> value) {psi.log_prefactor = complex_t(value);}
        )
        // return just a copy obviously, so writing has no effect..
        .def_property_readonly(
            "psi_ref",
            [](const PsiClassicalFP<2u>& psi) {return psi.psi_ref;}
        )
        .def_readonly("gpu", &PsiClassicalFP<2u>::gpu)
        .def_readonly("num_params", &PsiClassicalFP<2u>::num_params)
        .def_property_readonly("order", [](const PsiClassicalFP<2u>& psi) {return psi.get_order();})
        .def_property(
            "params",
            [](const PsiClassicalFP<2u>& psi) {return psi.params.to_pytensor_1d();},
            [](PsiClassicalFP<2u>& psi, const complex_tensor<1u>& new_params) {psi.params = new_params;}
        )
        .def("update_psi_ref_kernel", &PsiClassicalFP<2u>::update_psi_ref_kernel)
        #ifdef ENABLE_EXACT_SUMMATION
        #ifdef ENABLE_SPINS
        .def("vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_SPINS
        #ifdef ENABLE_PAULIS
        .def("vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_PAULIS
        #endif // ENABLE_EXACT_SUMMATION
        ;
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    py::class_<PsiClassicalANN<1u>>(m, "PsiClassicalANN_1")
        .def(py::init<
            const unsigned int,
            const vector<Operator_t>&,
            const vector<Operator_t>&,
            const complex_tensor<1u>&,
            const typename PsiClassicalANN<1u>::PsiRef&,
            const double,
            const bool
        >())
        .def("copy", &PsiClassicalANN<1u>::copy)
        .def("__pos__", &PsiClassicalANN<1u>::copy)
        .def_readwrite("num_sites", &PsiClassicalANN<1u>::num_sites)
        .def_readonly("H_local", &PsiClassicalANN<1u>::H_local)
        .def_readonly("H_2_local", &PsiClassicalANN<1u>::H_2_local)
        .def_readwrite("prefactor", &PsiClassicalANN<1u>::prefactor)
        .def_property(
            "log_prefactor",
            [](const PsiClassicalANN<1u>& psi) {return psi.log_prefactor.to_std();},
            [](PsiClassicalANN<1u>& psi, complex<double> value) {psi.log_prefactor = complex_t(value);}
        )
        // return just a copy obviously, so writing has no effect..
        .def_property_readonly(
            "psi_ref",
            [](const PsiClassicalANN<1u>& psi) {return psi.psi_ref;}
        )
        .def_readonly("gpu", &PsiClassicalANN<1u>::gpu)
        .def_readonly("num_params", &PsiClassicalANN<1u>::num_params)
        .def_property_readonly("order", [](const PsiClassicalANN<1u>& psi) {return psi.get_order();})
        .def_property(
            "params",
            [](const PsiClassicalANN<1u>& psi) {return psi.params.to_pytensor_1d();},
            [](PsiClassicalANN<1u>& psi, const complex_tensor<1u>& new_params) {psi.params = new_params;}
        )
        .def("update_psi_ref_kernel", &PsiClassicalANN<1u>::update_psi_ref_kernel)
        #ifdef ENABLE_EXACT_SUMMATION
        #ifdef ENABLE_SPINS
        .def("vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_SPINS
        #ifdef ENABLE_PAULIS
        .def("vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_PAULIS
        #endif // ENABLE_EXACT_SUMMATION
        ;
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    py::class_<PsiClassicalANN<2u>>(m, "PsiClassicalANN_2")
        .def(py::init<
            const unsigned int,
            const vector<Operator_t>&,
            const vector<Operator_t>&,
            const complex_tensor<1u>&,
            const typename PsiClassicalANN<2u>::PsiRef&,
            const double,
            const bool
        >())
        .def("copy", &PsiClassicalANN<2u>::copy)
        .def("__pos__", &PsiClassicalANN<2u>::copy)
        .def_readwrite("num_sites", &PsiClassicalANN<2u>::num_sites)
        .def_readonly("H_local", &PsiClassicalANN<2u>::H_local)
        .def_readonly("H_2_local", &PsiClassicalANN<2u>::H_2_local)
        .def_readwrite("prefactor", &PsiClassicalANN<2u>::prefactor)
        .def_property(
            "log_prefactor",
            [](const PsiClassicalANN<2u>& psi) {return psi.log_prefactor.to_std();},
            [](PsiClassicalANN<2u>& psi, complex<double> value) {psi.log_prefactor = complex_t(value);}
        )
        // return just a copy obviously, so writing has no effect..
        .def_property_readonly(
            "psi_ref",
            [](const PsiClassicalANN<2u>& psi) {return psi.psi_ref;}
        )
        .def_readonly("gpu", &PsiClassicalANN<2u>::gpu)
        .def_readonly("num_params", &PsiClassicalANN<2u>::num_params)
        .def_property_readonly("order", [](const PsiClassicalANN<2u>& psi) {return psi.get_order();})
        .def_property(
            "params",
            [](const PsiClassicalANN<2u>& psi) {return psi.params.to_pytensor_1d();},
            [](PsiClassicalANN<2u>& psi, const complex_tensor<1u>& new_params) {psi.params = new_params;}
        )
        .def("update_psi_ref_kernel", &PsiClassicalANN<2u>::update_psi_ref_kernel)
        #ifdef ENABLE_EXACT_SUMMATION
        #ifdef ENABLE_SPINS
        .def("vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_SPINS
        #ifdef ENABLE_PAULIS
        .def("vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_vector(psi, exact_summation).to_pytensor_1d();})
        .def("norm", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {return psi_norm(psi, exact_summation);})
        .def("normalize", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& exact_summation) {psi.prefactor = 1.0; psi.prefactor /= psi_norm(psi, exact_summation);})
        #endif // ENABLE_PAULIS
        #endif // ENABLE_EXACT_SUMMATION
        ;
#endif

#ifdef ENABLE_PSI_CLASSICAL
    py::class_<PsiFullyPolarized>(m, "PsiFullyPolarized")
        .def(py::init<unsigned int, double>())
        .def_readwrite("num_sites", &PsiFullyPolarized::num_sites);
#endif // ENABLE_PSI_CLASSICAL

#ifdef USE_SUPER_OPERATOR

    py::class_<SuperOperator>(m, "SuperOperator")
        .def(py::init<
            const vector<complex<double>>&,
            const vector<vector<unsigned int>>,
            const vector<vector<xt::pytensor<complex<double>, 2u>>>&,
            const bool
        >());

#else

    py::class_<Operator>(m, "Operator")
        .def(py::init<
            const quantum_expression::PauliExpression&,
            const bool
        >())
        .def_property_readonly("expr", &Operator::to_expr)
        .def_property_readonly("expr_list", &Operator::to_expr_list)
        .def_readonly("num_strings", &Operator::num_strings);

#endif // USE_SUPER_OPERATOR



#ifdef ENABLE_SPINS
    py::class_<ann_on_gpu::Spins>(m, "Spins")
        .def(py::init<ann_on_gpu::Spins::dtype, const unsigned int>())
        .def_static("enumerate", ann_on_gpu::Spins::enumerate)
        .def("roll", &ann_on_gpu::Spins::roll)
        .def("array", &ann_on_gpu::Spins::array);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
    py::class_<ann_on_gpu::PauliString>(m, "PauliString")
        .def(py::init<ann_on_gpu::PauliString::dtype, ann_on_gpu::PauliString::dtype>())
        .def_static("enumerate", ann_on_gpu::PauliString::enumerate)
        .def("roll", &ann_on_gpu::PauliString::roll)
        .def("array", &ann_on_gpu::PauliString::array);
#endif // ENABLE_PAULIS


#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_SPINS
    py::class_<MonteCarloSpins>(m, "MonteCarloSpins")
        .def(py::init(&make_MonteCarloSpins))
        .def(py::init<MonteCarloSpins&>())
        // .def("set_total_z_symmetry", &MonteCarloSpins::set_total_z_symmetry)
        .def_property_readonly("gpu", [](const MonteCarloSpins& ensemble) {return ensemble.gpu;})
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
        .def_property_readonly("gpu", [](const MonteCarloPaulis& ensemble) {return ensemble.gpu;})
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
        .def_property_readonly("gpu", [](const ExactSummationSpins& ensemble) {return ensemble.gpu;})
        // .def("set_total_z_symmetry", &ExactSummationSpins::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummationSpins::get_num_steps);
#endif // ENABLE_SPINS
#ifdef ENABLE_PAULIS
    py::class_<ExactSummationPaulis>(m, "ExactSummationPaulis")
        .def(py::init<unsigned int, bool>())
        .def_property_readonly("gpu", [](const ExactSummationPaulis& ensemble) {return ensemble.gpu;})
        // .def("set_total_z_symmetry", &ExactSummationPaulis::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummationPaulis::get_num_steps);
#endif // ENABLE_PAULIS
#endif // ENABLE_EXACT_SUMMATION


#ifndef LEAN_AND_MEAN

    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiFullyPolarized, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<1u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<2u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<1u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, MonteCarlo_tt<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<2u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiFullyPolarized, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<1u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<2u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<1u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, MonteCarlo_tt<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<2u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiFullyPolarized, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<1u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<2u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<1u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, ExactSummation_t<Spins>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<2u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiFullyPolarized, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiFullyPolarized, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<1u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalFP<2u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<1u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, ExactSummation_t<PauliString>>)
        .def("__call__", &ExpectationValue::__call__<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiClassicalANN<2u>, ExactSummation_t<PauliString>>)
#endif
    ;


    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, bool>())
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarlo_tt<Spins>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarlo_tt<Spins>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarlo_tt<PauliString>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarlo_tt<PauliString>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummation_t<Spins>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummation_t<Spins>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummation_t<PauliString>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummation_t<PauliString>>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif
    ;


    py::class_<KullbackLeibler>(m, "KullbackLeibler")
        .def(py::init<unsigned int, bool>())
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<1u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<2u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<1u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<2u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<1u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<2u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<1u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<1u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<2u>, PsiDeep, MonteCarlo_tt<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<2u>& psi, PsiDeep& psi_prime, MonteCarlo_tt<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<1u>& psi, PsiDeep& psi_prime, ExactSummation_t<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<2u>& psi, PsiDeep& psi_prime, ExactSummation_t<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<1u>& psi, PsiDeep& psi_prime, ExactSummation_t<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<Spins>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<2u>& psi, PsiDeep& psi_prime, ExactSummation_t<Spins>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<1u>& psi, PsiDeep& psi_prime, ExactSummation_t<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("__call__", &KullbackLeibler::value<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalFP<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalFP<2u>& psi, PsiDeep& psi_prime, ExactSummation_t<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<1u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<1u>& psi, PsiDeep& psi_prime, ExactSummation_t<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("__call__", &KullbackLeibler::value<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassicalANN<2u>, PsiDeep, ExactSummation_t<PauliString>>)
        .def("gradient_with_noise", [](KullbackLeibler& kl, PsiClassicalANN<2u>& psi, PsiDeep& psi_prime, ExactSummation_t<PauliString>& ensemble, const double nu, double threshold){
            const auto result = kl.gradient_with_noise(psi, psi_prime, ensemble, nu, threshold);
            return make_tuple(
                get<0>(result).to_pytensor_1d(),
                get<1>(result).to_pytensor_1d(),
                get<2>(result)
            );
        })
#endif
        .def_property_readonly("deviations", [](const KullbackLeibler& kl){return kl.deviations.to_pytensor_1d();})
    ;


    py::class_<TDVP>(m, "TDVP")
        .def(py::init<unsigned int, bool>())
        .def_property_readonly("S_matrix", [](const TDVP& tdvp){return tdvp.S_matrix.to_pytensor_2d({tdvp.num_params, tdvp.num_params});})
        .def_property_readonly("F_vector", [](const TDVP& tdvp){return tdvp.F_vector.to_pytensor_1d();})
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<1u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<2u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<1u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<2u>, MonteCarlo_tt<Spins>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<1u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<2u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<1u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<2u>, MonteCarlo_tt<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<1u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<2u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<1u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<2u>, ExactSummation_t<Spins>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<1u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
        .def("eval", &TDVP::eval<PsiClassicalFP<2u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<1u>, ExactSummation_t<PauliString>>)
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
        .def("eval", &TDVP::eval<PsiClassicalANN<2u>, ExactSummation_t<PauliString>>)
#endif
        ;

#endif // LEAN_AND_MEAN


#if defined(ENABLE_SPINS)
    m.def("log_psi_s", [](PsiDeep& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiDeep& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiFullyPolarized& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiFullyPolarized& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiClassicalFP<1u>& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalFP<1u>& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiClassicalFP<2u>& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalFP<2u>& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi_s", [](PsiClassicalANN<1u>& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalANN<1u>& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi_s", [](PsiClassicalANN<2u>& psi, const Spins& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalANN<2u>& psi, const Spins& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS)
    m.def("log_psi_s", [](PsiDeep& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiDeep& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiFullyPolarized& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiFullyPolarized& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiClassicalFP<1u>& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalFP<1u>& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi_s", [](PsiClassicalFP<2u>& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalFP<2u>& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi_s", [](PsiClassicalANN<1u>& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalANN<1u>& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi_s", [](PsiClassicalANN<2u>& psi, const PauliString& basis) {
        return log_psi_s(psi, basis);
    });
    m.def("psi_O_k", [](PsiClassicalANN<2u>& psi, const PauliString& basis) {
        return psi_O_k(psi, basis).to_pytensor_1d();
    });
#endif

    #ifdef ENABLE_EXACT_SUMMATION
#if defined(ENABLE_SPINS)
    m.def("psi_O_k_vector", [](PsiDeep& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("psi_O_k_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("psi_O_k_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS)
    m.def("psi_O_k_vector", [](PsiDeep& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("psi_O_k_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("psi_O_k_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("psi_O_k_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_O_k_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
    #endif // ENABLE_EXACT_SUMMATION

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
    m.def("log_psi", [](PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
    m.def("log_psi", [](PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
    m.def("log_psi", [](PsiDeep& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiDeep& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiDeep& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
    m.def("log_psi", [](PsiDeep& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiDeep& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiDeep& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("log_psi", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("log_psi", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi(psi, ensemble);
    });
    m.def("psi_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return psi_vector(psi, ensemble).to_pytensor_1d();
    });
    m.def("log_psi_vector", [](PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble) {
        return log_psi_vector(psi, ensemble).to_pytensor_1d();
    });
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
    m.def("apply_operator", [](PsiDeep& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiFullyPolarized& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<1u>& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<2u>& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<1u>& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<2u>& psi, const Operator_t& op, MonteCarlo_tt<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
    m.def("apply_operator", [](PsiDeep& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiFullyPolarized& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<1u>& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<2u>& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<1u>& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<2u>& psi, const Operator_t& op, MonteCarlo_tt<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
    m.def("apply_operator", [](PsiDeep& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiFullyPolarized& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<1u>& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<2u>& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<1u>& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<2u>& psi, const Operator_t& op, ExactSummation_t<Spins>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
    m.def("apply_operator", [](PsiDeep& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiFullyPolarized& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<1u>& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
    m.def("apply_operator", [](PsiClassicalFP<2u>& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<1u>& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
    m.def("apply_operator", [](PsiClassicalANN<2u>& psi, const Operator_t& op, ExactSummation_t<PauliString>& ensemble){
        return apply_operator(psi, op, ensemble).to_pytensor_1d();
    });
#endif


    py::class_<RNGStates>(m, "RNGStates")
        .def(py::init<unsigned int, bool>());


    m.def("activation_function", [](const complex<double>& x) {
        return my_logcosh(complex_t(x.real(), x.imag())).to_std();
    });

    m.def("deep_activation", [](const complex<double>& x) {
        return deep_activation(complex_t(x.real(), x.imag())).to_std();
    });

    m.def("setDevice", setDevice);
    m.def("start_profiling", start_profiling);
    m.def("stop_profiling", stop_profiling);
}
