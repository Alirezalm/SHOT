/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#include "Iteration.h"

Iteration::Iteration()
{
    this->iterationNumber = ProcessInfo::getInstance().iterations.size() + 1;

    this->numHyperplanesAdded = 0;

    if (ProcessInfo::getInstance().iterations.size() == 0)
        this->totNumHyperplanes = 0;
    else
        this->totNumHyperplanes = ProcessInfo::getInstance().iterations.at(ProcessInfo::getInstance().iterations.size() - 1).totNumHyperplanes;

    this->maxDeviation = OSDBL_MAX;
    this->boundaryDistance = OSDBL_MAX;
    this->MIPSolutionLimitUpdated = false;
    this->solutionStatus = E_ProblemSolutionStatus::None;

    currentObjectiveBounds.first = ProcessInfo::getInstance().getDualBound();
    currentObjectiveBounds.second = ProcessInfo::getInstance().getPrimalBound();

    if (ProcessInfo::getInstance().relaxationStrategy != NULL)
    {
        this->type = ProcessInfo::getInstance().relaxationStrategy->getProblemType();
    }
    else
    {
        switch (static_cast<E_SolutionStrategy>(ProcessInfo::getInstance().usedSolutionStrategy))
        {
        case (E_SolutionStrategy::MIQCQP):
            this->type = E_IterationProblemType::MIP;
            break;
        case (E_SolutionStrategy::MIQP):
            this->type = E_IterationProblemType::MIP;
            break;
        case (E_SolutionStrategy::NLP):
            this->type = E_IterationProblemType::Relaxed;
            break;
        case (E_SolutionStrategy::SingleTree):
            this->type = E_IterationProblemType::MIP;
            break;
        default:
            this->type = E_IterationProblemType::MIP;
            break;
        }
    }
}

Iteration::~Iteration()
{
    solutionPoints.clear();
    constraintDeviations.clear();
    hyperplanePoints.clear();
}

bool Iteration::isMIP()
{
    return (this->type == E_IterationProblemType::MIP);
}

SolutionPoint Iteration::getSolutionPointWithSmallestDeviation()
{
    double tmpVal = -OSDBL_MAX;
    int tmpIdx = 0;

    for (int i = 0; i < solutionPoints.size(); i++)
    {
        if (solutionPoints.at(i).maxDeviation.value > tmpVal)
        {
            tmpIdx = i;
            tmpVal = solutionPoints.at(i).maxDeviation.value;
        }
    }

    return (solutionPoints.at(tmpIdx));
}

int Iteration::getSolutionPointWithSmallestDeviationIndex()
{
    double tmpVal = -OSDBL_MAX;
    int tmpIdx = 0;

    for (int i = 0; i < solutionPoints.size(); i++)
    {
        if (solutionPoints.at(i).maxDeviation.value > tmpVal)
        {
            tmpIdx = i;
            tmpVal = solutionPoints.at(i).maxDeviation.value;
        }
    }

    return (tmpIdx);
}
