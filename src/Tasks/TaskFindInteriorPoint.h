/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "../NLPSolver/NLPSolverCuttingPlaneMinimax.h"
#include "TaskBase.h"
#include "../ProcessInfo.h"
#include "../NLPSolver/INLPSolver.h"
#include "../NLPSolver/NLPSolverIpoptMinimax.h"
#include "../NLPSolver/NLPSolverIpoptRelaxed.h"

class TaskFindInteriorPoint : public TaskBase
{
  public:
    TaskFindInteriorPoint(EnvironmentPtr envPtr);
    virtual ~TaskFindInteriorPoint();

    virtual void run();
    virtual std::string getType();

  private:
    std::vector<std::unique_ptr<INLPSolver>> NLPSolvers;
};