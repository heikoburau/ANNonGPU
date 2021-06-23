#pragma once



#ifdef USE_SUPER_OPERATOR

#include "operator/SuperOperator.hpp"

namespace ann_on_gpu {
    using Operator = SuperOperator;
} // namespace ann_on_gpu

#else

#include "operator/Operator.hpp"

namespace ann_on_gpu {
    using Operator = Operator;
} // namespace ann_on_gpu


#endif // USE_SUPER_OPERATOR
