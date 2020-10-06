#pragma once

#ifdef ENABLE_MONTE_CARLO
#include "ensembles/MonteCarloLoop.hpp"
#endif // ENABLE_MONTE_CARLO

#ifdef ENABLE_EXACT_SUMMATION
#include "ensembles/ExactSummation.hpp"
#endif // ENABLE_EXACT_SUMMATION

#ifdef ENABLE_MONTE_CARLO_PAULIS
#include "ensembles/MonteCarloLoopPaulis.hpp"
#endif // ENABLE_MONTE_CARLO_PAULIS
