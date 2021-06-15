#pragma once

#ifdef ENABLE_PSI_DEEP
#include "quantum_state/PsiDeep.hpp"
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_CNN
#include "quantum_state/PsiCNN.hpp"
#endif // ENABLE_PSI_CNN

#ifdef ENABLE_PSI_EXACT
#include "quantum_state/PsiExact.hpp"
#endif // ENABLE_PSI_EXACT

#ifdef ENABLE_PSI_CLASSICAL
#include "quantum_state/PsiFullyPolarized.hpp"
#include "quantum_state/PsiClassical.hpp"
#endif  // ENABLE_PSI_CLASSICAL

#ifdef ENABLE_PSI_RBM
#include "quantum_state/PsiRBM.hpp"
#endif // ENABLE_PSI_RBM
