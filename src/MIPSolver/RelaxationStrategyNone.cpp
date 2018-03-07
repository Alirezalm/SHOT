/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#include "RelaxationStrategyNone.h"

RelaxationStrategyNone::RelaxationStrategyNone(IMIPSolver *MIPSolver)
{
    this->MIPSolver = MIPSolver;
}

RelaxationStrategyNone::~RelaxationStrategyNone()
{
}

void RelaxationStrategyNone::setInitial()
{
    setInactive();
}

void RelaxationStrategyNone::executeStrategy()
{
    ProcessInfo::getInstance().getCurrentIteration()->type = E_IterationProblemType::MIP;
}

void RelaxationStrategyNone::setActive()
{
}

void RelaxationStrategyNone::setInactive()
{
}

E_IterationProblemType RelaxationStrategyNone::getProblemType()
{
    if (MIPSolver->getDiscreteVariableStatus())
        return E_IterationProblemType::MIP;
    else
        return E_IterationProblemType::Relaxed;
}