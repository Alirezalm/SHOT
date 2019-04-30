/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#pragma once

#include "IMIPSolutionLimitStrategy.h"
#include "Environment.h"

namespace SHOT
{
class MIPSolutionLimitStrategyIncrease : public IMIPSolutionLimitStrategy
{
public:
    MIPSolutionLimitStrategyIncrease(EnvironmentPtr envPtr);
    ~MIPSolutionLimitStrategyIncrease() = default;

    virtual bool updateLimit();
    virtual int getNewLimit();
    virtual int getInitialLimit();

    int lastIterSolLimIncreased;
    int numSolLimIncremented;
    int lastIterOptimal;
};
} // namespace SHOT