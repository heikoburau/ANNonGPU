from ._pyANNonGPU import *

from .PsiDeep import PsiDeep

from .new_neural_network import (
    new_deep_neural_network
)

from .new_classical_network import *

from .LearningByGradientDescent import LearningByGradientDescent, DidNotConverge
from .L2Regularization import L2Regularization

from .gradient_descent import *

try:
    from .SuperOperator import SuperOperator
except ImportError:
    pass
