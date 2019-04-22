/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#include "TaskCheckConstraintTolerance.h"

#include "../Enums.h"
#include "../Iteration.h"
#include "../Results.h"
#include "../Settings.h"
#include "../TaskHandler.h"
#include "../Timing.h"

#include "../Model/Problem.h"

namespace SHOT
{

TaskCheckConstraintTolerance::TaskCheckConstraintTolerance(EnvironmentPtr envPtr, std::string taskIDTrue)
    : TaskBase(envPtr), taskIDIfTrue(taskIDTrue)
{
}

TaskCheckConstraintTolerance::~TaskCheckConstraintTolerance() {}

void TaskCheckConstraintTolerance::run()
{
    auto currIter = env->results->getCurrentIteration();

    if(currIter->solutionPoints.size() == 0)
        return;

    if(env->reformulatedProblem->properties.isMIQPProblem || env->reformulatedProblem->properties.isQPProblem)
        return;

    auto constraintTolerance = env->settings->getSetting<double>("ConstraintTolerance", "Termination");

    // Checks it the nonlinear objective is fulfilled
    if(env->problem->objectiveFunction->properties.classification > E_ObjectiveFunctionClassification::Quadratic
        && env->problem->objectiveFunction->calculateValue(currIter->solutionPoints.at(0).point)
                - currIter->objectiveValue
            > constraintTolerance)
    {
        return;
    }

    // Checks if the quadratic constraints are fulfilled to tolerance
    if(!env->problem->areQuadraticConstraintsFulfilled(currIter->solutionPoints.at(0).point,
           env->settings->getSetting<double>("ConstraintTolerance", "Termination")))
    {
        return;
    }

    // Checks if the nonlinear constraints are fulfilled to tolerance
    if(!env->problem->areNonlinearConstraintsFulfilled(currIter->solutionPoints.at(0).point,
           env->settings->getSetting<double>("ConstraintTolerance", "Termination")))
    {
        return;
    }

    if(env->problem->properties.isDiscrete)
    {
        if(currIter->solutionStatus == E_ProblemSolutionStatus::Optimal
            && currIter->type == E_IterationProblemType::MIP)
        {
            env->results->terminationReason = E_TerminationReason::ConstraintTolerance;
            env->tasks->setNextTask(taskIDIfTrue);
            env->results->terminationReasonDescription = "Terminated since nonlinear constraint tolerance met.";
        }
    }
    else
    {
        if(currIter->solutionStatus == E_ProblemSolutionStatus::Optimal)
        {
            env->results->terminationReason = E_TerminationReason::ConstraintTolerance;
            env->tasks->setNextTask(taskIDIfTrue);
            env->results->terminationReasonDescription = "Terminated since nonlinear constraint tolerance met.";
        }
    }

    return;
}

std::string TaskCheckConstraintTolerance::getType()
{
    std::string type = typeid(this).name();
    return (type);
}
} // namespace SHOT