#pragma once


#ifdef ENABLE_SPINS
#include "basis/Spins.h"
#include "basis/PauliString.hpp"
#endif


#ifdef ENABLE_PAULIS
#include "basis/PauliString.hpp"
#endif


#ifdef ENABLE_FERMIONS
#include "basis/Fermions.hpp"
#include "basis/FermiString.hpp"
#endif
