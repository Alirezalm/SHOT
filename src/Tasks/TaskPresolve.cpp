/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#include "TaskPresolve.h"

namespace SHOT
{

TaskPresolve::TaskPresolve(EnvironmentPtr envPtr) : TaskBase(envPtr)
{
    env->timing->startTimer("DualStrategy");

    isPresolved = false;

    env->timing->stopTimer("DualStrategy");
}

TaskPresolve::~TaskPresolve() {}

void TaskPresolve::run()
{
    env->timing->startTimer("DualStrategy");
    auto currIter = env->results->getCurrentIteration();

    auto strategy = static_cast<ES_MIPPresolveStrategy>(env->settings->getIntSetting("MIP.Presolve.Frequency", "Dual"));

    if(!currIter->isMIP())
    {
        env->timing->stopTimer("DualStrategy");
        return;
    }

    if(strategy == ES_MIPPresolveStrategy::Never)
    {
        env->timing->stopTimer("DualStrategy");
        return;
    }
    else if(strategy == ES_MIPPresolveStrategy::Once && isPresolved == true)
    {
        env->timing->stopTimer("DualStrategy");
        return;
    }

    // Sets the iteration time limit
    auto timeLim = env->settings->getDoubleSetting("TimeLimit", "Termination") - env->timing->getElapsedTime("Total");
    env->dualSolver->MIPSolver->setTimeLimit(timeLim);

    if(env->results->primalSolutions.size() > 0)
    {
        // env->dualSolver->MIPSolver->setCutOff(env->results->getPrimalBound());
    }

    if(env->dualSolver->MIPSolver->getDiscreteVariableStatus() && env->results->primalSolutions.size() > 0)
    {
        env->dualSolver->MIPSolver->addMIPStart(env->results->primalSolution);
    }

    if(env->settings->getBoolSetting("FixedInteger.UsePresolveBounds", "Primal")
        || env->settings->getBoolSetting("MIP.Presolve.UpdateObtainedBounds", "Dual"))
    {
        env->dualSolver->MIPSolver->presolveAndUpdateBounds();
        isPresolved = true;
    }

    env->timing->stopTimer("DualStrategy");
}

std::string TaskPresolve::getType()
{
    std::string type = typeid(this).name();
    return (type);
}
} // namespace SHOT