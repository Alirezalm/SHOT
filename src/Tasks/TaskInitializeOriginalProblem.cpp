/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#include "TaskInitializeOriginalProblem.h"

TaskInitializeOriginalProblem::TaskInitializeOriginalProblem(EnvironmentPtr envPtr, OSInstance *originalInstance) : TaskBase(envPtr)
{
    env->process->startTimer("ProblemInitialization");

    // This is needed to fix various problems later on.
    // TODO: figure out why...
    originalInstance->getJacobianSparsityPattern();

    instance = originalInstance;

    bool useQuadraticObjective = (static_cast<ES_QuadraticProblemStrategy>(env->settings->getIntSetting("QuadraticStrategy", "Dual"))) == ES_QuadraticProblemStrategy::QuadraticObjective;

    bool useQuadraticConstraint = (static_cast<ES_QuadraticProblemStrategy>(env->settings->getIntSetting("QuadraticStrategy", "Dual"))) == ES_QuadraticProblemStrategy::QuadraticallyConstrained;

    bool isObjNonlinear = UtilityFunctions::isObjectiveGenerallyNonlinear(instance);
    bool isObjQuadratic = UtilityFunctions::isObjectiveQuadratic(instance);
    bool isQuadraticUsed = (useQuadraticObjective || (useQuadraticConstraint));

    if (isObjNonlinear || (isObjQuadratic && !isQuadraticUsed))
    {
        env->output->outputInfo("Nonlinear objective function detected.");
        env->process->originalProblem = new OptProblemOriginalNonlinearObjective(env);
    }
    else if (isObjQuadratic && isQuadraticUsed)
    {
        env->output->outputInfo("Quadratic objective function detected.");
        env->process->originalProblem = new OptProblemOriginalQuadraticObjective(env);
    }
    else //Linear objective function
    {
        env->output->outputInfo("Linear objective function detected.");
        env->process->originalProblem = new OptProblemOriginalLinearObjective(env);
    }

    env->process->originalProblem->setProblem(instance);
    auto debugPath = env->settings->getStringSetting("Debug.Path", "Output");

    if (env->settings->getBoolSetting("Debug.Enable", "Output"))
    {
        env->process->originalProblem->saveProblemModelToFile(
            env->settings->getStringSetting("Debug.Path", "Output") + "/originalproblem.txt");
    }

    int numConstr = env->process->originalProblem->getNumberOfConstraints();

    int numVar = env->process->originalProblem->getNumberOfVariables();

    if (env->process->originalProblem->isObjectiveFunctionNonlinear())
    {
        numVar = numVar - 1;       // Removes the extra objective variable
        numConstr = numConstr - 1; // Removes the extra objective constraint
    }

    env->process->initializeResults(1, numVar, numConstr);

    env->process->stopTimer("ProblemInitialization");
}

TaskInitializeOriginalProblem::~TaskInitializeOriginalProblem()
{
    //delete problem;
    // delete instance;
}

void TaskInitializeOriginalProblem::run()
{
}

std::string TaskInitializeOriginalProblem::getType()
{
    std::string type = typeid(this).name();
    return (type);
}
