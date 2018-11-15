/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#include "TaskSelectHyperplanePointsLinesearch.h"

namespace SHOT
{

TaskSelectHyperplanePointsLinesearch::TaskSelectHyperplanePointsLinesearch(EnvironmentPtr envPtr) : TaskBase(envPtr)
{
    env->process->startTimer("DualCutGenerationRootSearch");
    env->process->stopTimer("DualCutGenerationRootSearch");
}

TaskSelectHyperplanePointsLinesearch::~TaskSelectHyperplanePointsLinesearch()
{
    if (hyperplaneSolutionPointStrategyInitialized)
    {
        delete tSelectHPPts;
    }
}

void TaskSelectHyperplanePointsLinesearch::run()
{
    this->run(env->process->getPreviousIteration()->solutionPoints);
}

void TaskSelectHyperplanePointsLinesearch::run(std::vector<SolutionPoint> solPoints)
{
    env->process->startTimer("DualCutGenerationRootSearch");

    int addedHyperplanes = 0;

    auto currIter = env->process->getCurrentIteration(); // The unsolved new iteration

    if (env->process->interiorPts.size() == 0)
    {
        if (!hyperplaneSolutionPointStrategyInitialized)
        {
            tSelectHPPts = new TaskSelectHyperplanePointsSolution(env);
            hyperplaneSolutionPointStrategyInitialized = true;
        }

        env->output->outputWarning("     Adding cutting plane since no interior point is known.");
        tSelectHPPts->run(solPoints);

        env->process->stopTimer("DualCutGenerationRootSearch");
        return;
    }

    for (int i = 0; i < solPoints.size(); i++)
    {
        auto maxDevConstr = env->reformulatedProblem->getMostDeviatingNonlinearConstraint(solPoints.at(i).point);

        if (maxDevConstr)
        {
            for (int j = 0; j < env->process->interiorPts.size(); j++)
            {
                if (addedHyperplanes >= env->settings->getIntSetting("HyperplaneCuts.MaxPerIteration", "Dual"))
                {
                    env->process->stopTimer("DualCutGenerationRootSearch");
                    return;
                }

                auto xNLP = env->process->interiorPts.at(j)->point;

                if (isnan(maxDevConstr->functionValue))
                {
                    auto t = maxDevConstr.get();
                    continue;
                }

                /*
                // Do not add hyperplane if less than this tolerance or negative
                if (maxDevConstr->error < env->settings->getDoubleSetting("ESH.Linesearch.ConstraintTolerance", "Dual"))
                    continue;
                */

                // Do not add hyperplane if constraint value is much less than largest
                if (maxDevConstr->error < env->settings->getDoubleSetting("ESH.Linesearch.ConstraintFactor", "Dual") * maxDevConstr.get().error)
                    continue;

                VectorDouble externalPoint;
                VectorDouble internalPoint;

                try
                {
                    env->process->startTimer("DualCutGenerationRootSearch");
                    auto xNewc = env->process->linesearchMethod->findZero(xNLP, solPoints.at(i).point,
                                                                          env->settings->getIntSetting("Rootsearch.MaxIterations", "Subsolver"),
                                                                          env->settings->getDoubleSetting("Rootsearch.TerminationTolerance", "Subsolver"),
                                                                          env->settings->getDoubleSetting("Rootsearch.ActiveConstraintTolerance", "Subsolver"),
                                                                          env->reformulatedProblem->nonlinearConstraints);

                    env->process->stopTimer("DualCutGenerationRootSearch");
                    internalPoint = xNewc.first;
                    externalPoint = xNewc.second;
                }
                catch (std::exception &e)
                {
                    env->process->stopTimer("DualCutGenerationRootSearch");
                    externalPoint = solPoints.at(i).point;

                    env->output->outputWarning(
                        "     Cannot find solution with linesearch, using solution point instead.");
                }

                std::vector<NonlinearConstraint *> activeConstraints;
                auto maxConstraintValue = env->reformulatedProblem->getMaxNumericConstraintValue(externalPoint, env->reformulatedProblem->nonlinearConstraints, activeConstraints);

                if (maxConstraintValue.normalizedValue >= 0)
                {
                    Hyperplane hyperplane;
                    hyperplane.sourceConstraint = maxConstraintValue.constraint;
                    hyperplane.sourceConstraintIndex = maxConstraintValue.constraint->index;
                    hyperplane.generatedPoint = externalPoint;

                    if (solPoints.at(i).isRelaxedPoint)
                    {
                        hyperplane.source = E_HyperplaneSource::MIPCallbackRelaxed;
                    }
                    else if (i == 0 && currIter->isMIP())
                    {
                        hyperplane.source = E_HyperplaneSource::MIPOptimalLinesearch;
                    }
                    else if (currIter->isMIP())
                    {
                        hyperplane.source = E_HyperplaneSource::MIPSolutionPoolLinesearch;
                    }
                    else
                    {
                        hyperplane.source = E_HyperplaneSource::LPRelaxedLinesearch;
                    }

                    env->process->hyperplaneWaitingList.push_back(hyperplane);
                    addedHyperplanes++;

                    env->output->outputInfo(
                        "     Added hyperplane to waiting list with deviation: " + UtilityFunctions::toString(maxConstraintValue.error));
                }
                else
                {
                    env->output->outputAlways("     Could not add hyperplane to waiting list.");
                }
            }
        }
    }

    env->process->stopTimer("DualCutGenerationRootSearch");
}

std::string TaskSelectHyperplanePointsLinesearch::getType()
{
    std::string type = typeid(this).name();
    return (type);
}
} // namespace SHOT