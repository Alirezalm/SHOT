/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "TaskBase.h"
#include "../ProcessInfo.h"
#include "../MIPSolver/IMIPSolutionLimitStrategy.h"
#include "../MIPSolver/MIPSolutionLimitStrategyUnlimited.h"
#include "../MIPSolver/MIPSolutionLimitStrategyIncrease.h"
#include "../MIPSolver/MIPSolutionLimitStrategyAdaptive.h"

class TaskExecuteSolutionLimitStrategy : public TaskBase
{
  public:
    TaskExecuteSolutionLimitStrategy(IMIPSolver *MIPSolver);
    virtual ~TaskExecuteSolutionLimitStrategy();

    void run();
    virtual std::string getType();

  private:
    IMIPSolutionLimitStrategy *solutionLimitStrategy;

    IMIPSolver *MIPSolver;

    bool isInitialized;
    bool temporaryOptLimitUsed;
    int previousSolLimit;
};
