/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "IMIPSolutionLimitStrategy.h"

class MIPSolutionLimitStrategyUnlimited : public IMIPSolutionLimitStrategy
{
  public:
	MIPSolutionLimitStrategyUnlimited(IMIPSolver *MIPSolver);
	~MIPSolutionLimitStrategyUnlimited();

	virtual bool updateLimit();
	virtual int getNewLimit();
	virtual int getInitialLimit();
};
