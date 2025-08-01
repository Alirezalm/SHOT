/** getCurrentDualBound() getCurrentDualBound() getCurrentDualBound()
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#include "TaskParallelSelectPrimalCandidatesFromNLP.h"
#include <thread>
#include "../DualSolver.h"
#include "../MIPSolver/IMIPSolver.h"
#include "../Output.h"
#include "../PrimalSolver.h"
#include "../Report.h"
#include "../Results.h"
#include "../Settings.h"
#include "../Solver.h"
#include "../Timing.h"
#include "../Utilities.h"
#include "../Concurrency/SHOTThreadPool.h"

#include "../Model/Problem.h"
#include "../NLPSolver/INLPSolver.h"

#include "../Tasks/TaskSelectHyperplanePointsESH.h"
#include "../Tasks/TaskSelectHyperplanePointsECP.h"
#include "Structs.h"
#include <mutex>

#ifdef HAS_IPOPT
#include "../NLPSolver/NLPSolverIpoptRelaxed.h"
#endif

#ifdef HAS_GAMS
#include "../NLPSolver/NLPSolverGAMS.h"
#include "../ModelingSystem/ModelingSystemGAMS.h"
#endif

#include "../NLPSolver/NLPSolverSHOT.h"
#include "cppad/utility/thread_alloc.hpp"

namespace SHOT
{

thread_local size_t tl_thread_num = 0;
thread_local VectorInteger discreteVariableIndexes;
thread_local VectorString variableNames;
thread_local ProblemPtr sourceProblem;

std::atomic<bool> parallel_mode { false };
bool in_parallel() { return parallel_mode.load(); }
size_t thread_num() { return tl_thread_num; }

NLPSolverPtr TaskParallelSelectPrimalCandidatesFromNLP::createNLPSolver(bool useReformulatedProblem)
{
    discreteVariableIndexes.clear();
    variableNames.clear();

    NLPSolverPtr nlpSolver;

    auto nlpSolverType
        = static_cast<ES_PrimalNLPSolver>(env->settings->getSetting<int>("FixedInteger.Solver", "Primal"));

    switch(nlpSolverType)
    {
#ifdef HAS_IPOPT
    case(ES_PrimalNLPSolver::Ipopt):
    {
        if(useReformulatedProblem)
        {
            sourceProblem = env->reformulatedProblem->createCopy(env);
            sourceIsReformulatedProblem = true;
        }
        else
        {
            sourceProblem = env->problem->createCopy(env);
            sourceIsReformulatedProblem = false;
        }

        env->results->usedPrimalNLPSolver = ES_PrimalNLPSolver::Ipopt;
        nlpSolver = std::make_shared<NLPSolverIpoptRelaxed>(env, sourceProblem);
        break;
    }
#endif

#ifdef HAS_GAMS
    case(ES_PrimalNLPSolver::GAMS):
    {
        // GAMS only has the original problem model
        sourceProblem = env->problem->createCopy(env);
        sourceIsReformulatedProblem = false;

        env->results->usedPrimalNLPSolver = ES_PrimalNLPSolver::GAMS;
        nlpSolver = std::make_shared<NLPSolverGAMS>(env,
            (std::dynamic_pointer_cast<ModelingSystemGAMS>(env->modelingSystem))->modelingObject,
            (std::dynamic_pointer_cast<ModelingSystemGAMS>(env->modelingSystem))->auditLicensing);

        break;
    }
#endif

    case(ES_PrimalNLPSolver::SHOT):
    {
        // Always use the reformulated problem with SHOT
        sourceProblem = env->reformulatedProblem->createCopy(env);

        env->results->usedPrimalNLPSolver = ES_PrimalNLPSolver::SHOT;
        nlpSolver = std::make_shared<NLPSolverSHOT>(env, sourceProblem);
        sourceIsReformulatedProblem = true;

        break;
    }

    default:
        // We should never get here since there is a check in Solver.cpp that makes sure that the correct solver is used
        break;
    }

    env->results->usedPrimalNLPSolverDescription
        = nlpSolver->getSolverDescription(); // todo: should be moved to the constructor

    this->originalIterFrequency = env->settings->getSetting<int>("FixedInteger.Frequency.Iteration", "Primal");
    this->originalTimeFrequency = env->settings->getSetting<double>("FixedInteger.Frequency.Time", "Primal");

    for(auto& V : sourceProblem->binaryVariables)
    {
        discreteVariableIndexes.push_back(V->index);
    }

    for(auto& V : sourceProblem->integerVariables)
    {
        discreteVariableIndexes.push_back(V->index);
    }

    for(auto& V : sourceProblem->semiintegerVariables)
    {
        discreteVariableIndexes.push_back(V->index);
    }

    if(env->settings->getSetting<bool>("Debug.Enable", "Output"))
    {
        for(auto& V : sourceProblem->allVariables)
        {
            variableNames.push_back(V->name);
        }
    }

    for(auto& V : sourceProblem->allVariables)
    {
        nlpSolver->updateVariableLowerBound(V->index, V->lowerBound);
        nlpSolver->updateVariableUpperBound(V->index, V->upperBound);
    }

    return nlpSolver;
}

TaskParallelSelectPrimalCandidatesFromNLP::TaskParallelSelectPrimalCandidatesFromNLP(
    EnvironmentPtr envPtr, bool useReformulatedProblem)
    : TaskBase(envPtr), useReformulatedProblem(useReformulatedProblem)
{
    size_t numThreads = env->parallelSHOT->getThreadCount();

    CppAD::thread_alloc::parallel_setup(numThreads, in_parallel, thread_num);

    env->timing->startTimer("PrimalStrategy");
    env->timing->startTimer("PrimalBoundStrategyNLP");

    originalNLPTime = env->settings->getSetting<double>("FixedInteger.Frequency.Time", "Primal");
    originalNLPIter = env->settings->getSetting<int>("FixedInteger.Frequency.Iteration", "Primal");

    env->timing->stopTimer("PrimalBoundStrategyNLP");
    env->timing->stopTimer("PrimalStrategy");
}

TaskParallelSelectPrimalCandidatesFromNLP::~TaskParallelSelectPrimalCandidatesFromNLP(){

    auto numThreads = env->parallelSHOT->getThreadCount();
    CppAD::thread_alloc::parallel_setup(numThreads, nullptr, nullptr);
};

void TaskParallelSelectPrimalCandidatesFromNLP::run()
{
    if(env->primalSolver->fixedPrimalNLPCandidates.size() == 0)
    {
        env->solutionStatistics.numberOfIterationsWithoutNLPCallMIP++;
        return;
    }

    if(env->results->getRelativeGlobalObjectiveGap() < 1e-10)
    {
        env->solutionStatistics.numberOfIterationsWithoutNLPCallMIP++;
        return;
    }

    env->timing->startTimer("PrimalStrategy");
    env->timing->startTimer("PrimalBoundStrategyNLP");

    parallelSolveFixedNLP();

    env->timing->stopTimer("PrimalBoundStrategyNLP");
    env->timing->stopTimer("PrimalStrategy");

    return;
}

std::string TaskParallelSelectPrimalCandidatesFromNLP::getType()
{
    std::string type = typeid(this).name();
    return (type);
}

bool TaskParallelSelectPrimalCandidatesFromNLP::parallelSolveFixedNLP()
{

    // std::vector<PrimalFixedNLPCandidate> testPts;

    // env->output->outputDebug("        Solving fixed NLP problem:");

    if(env->primalSolver->fixedPrimalNLPCandidates.size() == 0)
    {
        // env->output->outputDebug("         No candidate points available.");
        env->solutionStatistics.numberOfIterationsWithoutNLPCallMIP++;
        return (false);
    }

    int counter = 0;

    CppAD::thread_alloc::hold_memory(true);
    CppAD::parallel_ad<double>();
    parallel_mode.store(true);

    size_t i = 0;
    fmt::print("total number of candidates: {}\n", env->primalSolver->fixedPrimalNLPCandidates.size());

    for(auto& CAND : env->primalSolver->fixedPrimalNLPCandidates)
    {
        ++i;

        env->parallelSHOT->submitTask([this, CAND, i]() { processCandidate(CAND, i); });

        counter++;
    }

    env->parallelSHOT->waitForAllTasks();

    parallel_mode.store(false);
    CppAD::thread_alloc::hold_memory(false);
    CppAD::parallel_ad<double>();

    return (true);
}

void TaskParallelSelectPrimalCandidatesFromNLP::createInfeasibilityCut(const VectorDouble variableSolution)
{
    env->output->outputDebug("         Adding infeasibility cut from fixed NLP solution.");

    SolutionPoint tmpSolPt;
    tmpSolPt.point = variableSolution;
    tmpSolPt.objectiveValue = sourceProblem->objectiveFunction->calculateValue(variableSolution);
    tmpSolPt.iterFound = env->results->getCurrentIteration()->iterationNumber;

    if(auto mostDevConstr = sourceProblem->getMostDeviatingNonlinearOrQuadraticConstraint(variableSolution);
        mostDevConstr)
        tmpSolPt.maxDeviation = PairIndexValue(mostDevConstr->constraint->index, mostDevConstr->normalizedValue);

    if(!sourceIsReformulatedProblem) // Need to calculate values for the auxiliary variables in this
                                     // case
    {
        if((int)tmpSolPt.point.size() < env->reformulatedProblem->properties.numberOfVariables)
            env->reformulatedProblem->augmentAuxiliaryVariableValues(tmpSolPt.point);

        assert(tmpSolPt.point.size() == env->reformulatedProblem->properties.numberOfVariables);
    }

    std::vector<SolutionPoint> solutionPoints(1);
    solutionPoints.at(0) = tmpSolPt;

    if(!taskSelectHPPts)
    {
        if(static_cast<ES_HyperplaneCutStrategy>(env->settings->getSetting<int>("CutStrategy", "Dual"))
            == ES_HyperplaneCutStrategy::ESH)
            taskSelectHPPts = std::make_shared<TaskSelectHyperplanePointsESH>(env);
        else
            taskSelectHPPts = std::make_shared<TaskSelectHyperplanePointsECP>(env);
    }

    if(static_cast<ES_HyperplaneCutStrategy>(env->settings->getSetting<int>("CutStrategy", "Dual"))
        == ES_HyperplaneCutStrategy::ESH)
    {
        std::dynamic_pointer_cast<TaskSelectHyperplanePointsESH>(taskSelectHPPts)->run(solutionPoints);
    }
    else
    {
        std::dynamic_pointer_cast<TaskSelectHyperplanePointsECP>(taskSelectHPPts)->run(solutionPoints);
    }
}

void TaskParallelSelectPrimalCandidatesFromNLP::createIntegerCut(VectorDouble variableSolution)
{
    bool withinBounds = true;

    assert(variableSolution.size() == env->reformulatedProblem->variableLowerBounds.size());
    assert(variableSolution.size() == env->reformulatedProblem->variableUpperBounds.size());

    // Verify that solution is within bounds: if the difference is small project to the bound, otherwise do not add
    // integer cut
    for(size_t i = 0; i < variableSolution.size(); i++)
    {
        if(variableSolution[i] < env->reformulatedProblem->variableLowerBounds[i])
        {
            if(variableSolution[i] > env->reformulatedProblem->variableLowerBounds[i] - 1e-8)
                variableSolution[i] = env->reformulatedProblem->variableLowerBounds[i];
            else
            {
                withinBounds = false;
                break;
            }
        }

        if(variableSolution[i] > env->reformulatedProblem->variableUpperBounds[i])
        {
            if(variableSolution[i] < env->reformulatedProblem->variableUpperBounds[i] + 1e-8)
                variableSolution[i] = env->reformulatedProblem->variableUpperBounds[i];
            else
            {
                withinBounds = false;
                break;
            }
        }
    }

    if(!withinBounds)
    {
        env->output->outputDebug("         Can not add integer cut since solution is not within variable bounds.");
        return;
    }

    IntegerCut integerCut;
    integerCut.source = E_IntegerCutSource::NLPFixedInteger;
    integerCut.variableValues.reserve(discreteVariableIndexes.size());

    integerCut.variableIndexes = discreteVariableIndexes;

    for(auto& I : discreteVariableIndexes)
        integerCut.variableValues.push_back(round(variableSolution.at(I)));

    env->dualSolver->addIntegerCut(integerCut);
}

void TaskParallelSelectPrimalCandidatesFromNLP::processCandidate(PrimalFixedNLPCandidate CAND, size_t i)
{
    tl_thread_num = i;

    auto NLPSolver = createNLPSolver(useReformulatedProblem); // local

    VectorDouble fixedVariableValues(discreteVariableIndexes.size()); // local

    int sizeOfVariableVector = sourceProblem->properties.numberOfVariables; // local

    // TODO: remove?
    // if(env->settings->getSetting<bool>("FixedInteger.UsePresolveBounds", "Primal"))
    // {
    //     // env->output->outputDebug("         Updating variable bounds from MIP presolve.");
    //     for(auto& V : env->reformulatedProblem->allVariables)
    //     {
    //         if(V->index > sizeOfVariableVector)
    //             continue;

    //         if(V->properties.hasUpperBoundBeenTightened)
    //         {
    //             NLPSolver->updateVariableUpperBound(V->index, V->upperBound);
    //         }

    //         if(V->properties.hasLowerBoundBeenTightened)
    //         {
    //             NLPSolver->updateVariableLowerBound(V->index, V->upperBound);
    //         }
    //     }
    // }

    VectorInteger startingPointIndexes(sizeOfVariableVector); // local
    VectorDouble startingPointValues(sizeOfVariableVector); // local

    // Sets the fixed values for discrete variables
    for(size_t k = 0; k < discreteVariableIndexes.size(); k++)
    {
        int currVarIndex = discreteVariableIndexes.at(k);

        auto tmpSolPt = std::round(CAND.point.at(currVarIndex));

        fixedVariableValues.at(k) = tmpSolPt;

        // Sets the starting point to the fixed value
        if(env->settings->getSetting<bool>("FixedInteger.Warmstart", "Primal"))
        {
            startingPointIndexes.at(currVarIndex) = currVarIndex;
            startingPointValues.at(currVarIndex) = tmpSolPt;
        }
    }

    if(env->settings->getSetting<bool>("FixedInteger.Warmstart", "Primal"))
    {
        // env->output->outputDebug(
        //     "         Setting warm start for continuous variable to candidate solution value.");

        for(auto& V : sourceProblem->realVariables)
        {
            startingPointIndexes.at(V->index) = V->index;
            startingPointValues.at(V->index) = CAND.point.at(V->index);
        }

        // if(env->settings->getSetting<bool>("Debug.Enable", "Output"))
        // {
        //     auto filename = fmt::format("{}/primalnlp{}_warmstart_{}.txt",
        //         env->settings->getSetting<std::string>("Debug.Path", "Output"),
        //         env->results->getCurrentIteration()->iterationNumber - 1, counter);

        //     Utilities::saveVariablePointVectorToFile(startingPointValues, variableNames, filename);
        // }

        NLPSolver->setStartingPoint(startingPointIndexes, startingPointValues); // local
    }

    NLPSolver->fixVariables(discreteVariableIndexes, fixedVariableValues);

    // if(env->settings->getSetting<bool>("Debug.Enable", "Output"))
    // {
    //     std::string filename = env->settings->getSetting<std::string>("Debug.Path", "Output") + "/primalnlp"
    //         + std::to_string(currIter->iterationNumber) + "_" + std::to_string(counter);
    //     NLPSolver->saveProblemToFile(filename + ".txt");
    //     NLPSolver->saveOptionsToFile(filename + ".osrl");
    // }

    auto solvestatus = NLPSolver->solveProblem();
    auto currIter = env->results->getCurrentIteration();

    NLPSolver->unfixVariables(); // local uses ipopt problem internally which is shared.

    {
        std::lock_guard<std::mutex> lock(mutex);
        env->solutionStatistics.numberOfProblemsFixedNLP++;
    }

    std::string source = (sourceIsReformulatedProblem) ? "R" : "O"; // local

    std::string sourceDesc; // local
    switch(CAND.sourceType)
    {
    case E_PrimalNLPSource::FirstSolution:
        // env->output->outputDebug("         Source from candidate point is first MIP solution point.");
        sourceDesc = "SOLPT-" + source;
        break;
    case E_PrimalNLPSource::FeasibleSolution:
        // env->output->outputDebug("         Source from candidate point is MIP solution pool.");
        sourceDesc = "FEASP-" + source;
        break;
    case E_PrimalNLPSource::InfeasibleSolution:
        // env->output->outputDebug("         Source from candidate point is infeasible MIP solution.");
        sourceDesc = "UNFEA-" + source;
        break;
    case E_PrimalNLPSource::SmallestDeviationSolution:
        // env->output->outputDebug(
        //     "         Source from candidate point is MIP solution with smallest nonlinear error.");
        sourceDesc = "SMDEV-" + source;
        break;
    case E_PrimalNLPSource::FirstSolutionNewDualBound:
        // env->output->outputDebug(
        //     "         Source from candidate point is first MIP solution point which gave dual bound update.");
        sourceDesc = "NEWDB-" + source;
        break;
    default:
        break;
    }

    // switch(solvestatus)
    // {
    // case E_NLPSolutionStatus::Optimal:
    //     env->output->outputDebug(fmt::format(
    //         "         Optimal solution {} found to fixed NLP problem.", NLPSolver->getObjectiveValue()));
    //     break;

    // case E_NLPSolutionStatus::Feasible:
    //     env->output->outputDebug(fmt::format(
    //         "         Feasible solution {} found to fixed NLP problem.", NLPSolver->getObjectiveValue()));
    //     break;

    // case E_NLPSolutionStatus::Infeasible:
    //     env->output->outputDebug("         Fixed NLP problem is infeasible.");
    //     break;

    // case E_NLPSolutionStatus::Unbounded:
    //     env->output->outputDebug("         Fixed NLP problem is unbounded.");
    //     break;

    // case E_NLPSolutionStatus::TimeLimit:
    //     env->output->outputDebug("         Time limit hit when solving fixed NLP problem.");
    //     break;

    // case E_NLPSolutionStatus::IterationLimit:
    //     env->output->outputDebug("         Iteration limit hit when solving fixed NLP problem.");
    //     break;

    // case E_NLPSolutionStatus::Error:
    //     env->output->outputDebug("         Error ocurred when solving fixed NLP problem.");
    //     break;

    // default:

    //     break;
    // }

    if(solvestatus == E_NLPSolutionStatus::Feasible || solvestatus == E_NLPSolutionStatus::Optimal)
    {
        double tmpObj = NLPSolver->getObjectiveValue(); // local
        auto variableSolution = NLPSolver->getSolution(); // local

        {
            std::lock_guard<std::mutex> lock(mutex);
            env->primalSolver->addPrimalSolutionCandidate(
                variableSolution, E_PrimalSolutionSource::NLPFixedIntegers, currIter->iterationNumber);
        }

        if(sourceProblem->properties.numberOfNonlinearConstraints > 0
            || sourceProblem->properties.numberOfQuadraticConstraints > 0)
        {
            auto mostDevConstr
                = sourceProblem->getMostDeviatingNonlinearOrQuadraticConstraint(variableSolution); // local

            // env->output->outputDebug(fmt::format("         Max error {} from nonlinear or quadratic constraint {}.",
            //     mostDevConstr->normalizedValue, mostDevConstr->constraint->name));

            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), tmpObj,
                mostDevConstr->constraint->index, mostDevConstr->normalizedValue, E_IterationLineType::PrimalNLP);
        }
        else
        {
            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), tmpObj,
                -1, // Not shown
                0.0, // Not shown
                E_IterationLineType::PrimalNLP);
        }

        // Add integer cut.
        if(env->settings->getSetting<bool>("HyperplaneCuts.UseIntegerCuts", "Dual")
            && sourceProblem->properties.numberOfDiscreteVariables > 0)
        {
            std::lock_guard<std::mutex> lock(mutex);
            createIntegerCut(CAND.point);
        }

        if(env->settings->getSetting<bool>("FixedInteger.CreateInfeasibilityCut", "Primal"))
        {
            std::lock_guard<std::mutex> lock(mutex);
            createInfeasibilityCut(variableSolution);
        }
    }
    else if(solvestatus == E_NLPSolutionStatus::Error || solvestatus == E_NLPSolutionStatus::Unbounded
        || solvestatus == E_NLPSolutionStatus::Infeasible)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), NAN, -1,
                NAN, E_IterationLineType::PrimalNLP);
        }
    }
    else if(sourceProblem->properties.numberOfNonlinearConstraints > 0
        || sourceProblem->properties.numberOfQuadraticConstraints > 0)
    {
        double tmpObj = NLPSolver->getObjectiveValue(); // local

        auto variableSolution = NLPSolver->getSolution(); // local

        if(variableSolution.size() > 0)
        {
            auto mostDevConstr
                = sourceProblem->getMostDeviatingNonlinearOrQuadraticConstraint(variableSolution); // local

            if(env->settings->getSetting<bool>("FixedInteger.CreateInfeasibilityCut", "Primal"))
            {
                std::lock_guard<std::mutex> lock(mutex);
                createInfeasibilityCut(variableSolution);
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP,
                    ("NLP" + sourceDesc), env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded,
                    currIter->totNumHyperplanes, env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                    env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(),
                    tmpObj, mostDevConstr->constraint->index, mostDevConstr->normalizedValue,
                    E_IterationLineType::PrimalNLP);
            }
        }
        else
        {
            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), NAN, -1,
                NAN, E_IterationLineType::PrimalNLP);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), NAN, -1,
                NAN, E_IterationLineType::PrimalNLP);
        }
    }
    else
    {

        auto variableSolution = NLPSolver->getSolution();

        if(variableSolution.size() > 0)
        {
            std::lock_guard<std::mutex> lock(mutex);
            env->report->outputIterationDetail(env->solutionStatistics.numberOfProblemsFixedNLP, ("NLP" + sourceDesc),
                env->timing->getElapsedTime("Total"), currIter->numHyperplanesAdded, currIter->totNumHyperplanes,
                env->results->getCurrentDualBound(), env->results->getPrimalBound(),
                env->results->getAbsoluteGlobalObjectiveGap(), env->results->getRelativeGlobalObjectiveGap(), NAN, -1,
                NAN, E_IterationLineType::PrimalNLP);
        }
    }

    if(env->settings->getSetting<bool>("FixedInteger.Frequency.Dynamic", "Primal"))
    {
        if(solvestatus == E_NLPSolutionStatus::Optimal || solvestatus == E_NLPSolutionStatus::Feasible)
        {
            std::lock_guard<std::mutex> lock(mutex);
            int iters = std::max(
                std::ceil(env->settings->getSetting<int>("FixedInteger.Frequency.Iteration", "Primal") * 0.98),
                originalNLPIter);

            if(iters > std::max(0.1 * this->originalIterFrequency, 1.0))
                env->settings->updateSetting("FixedInteger.Frequency.Iteration", "Primal", iters);

            double interval = std::max(
                0.9 * env->settings->getSetting<double>("FixedInteger.Frequency.Time", "Primal"), originalNLPTime);

            if(interval > 0.1 * this->originalTimeFrequency)
                env->settings->updateSetting("FixedInteger.Frequency.Time", "Primal", interval);

            env->output->outputDebug(fmt::format(
                "         Iteration frequency updated to {} and time frequency updated to {} ", iters, interval));
        }
        else
        {
            std::lock_guard<std::mutex> lock(mutex);
            int iters = std::ceil(env->settings->getSetting<int>("FixedInteger.Frequency.Iteration", "Primal") * 1.02);

            if(iters < 10 * this->originalIterFrequency)
                env->settings->updateSetting("FixedInteger.Frequency.Iteration", "Primal", iters);

            double interval = 1.1 * env->settings->getSetting<double>("FixedInteger.Frequency.Time", "Primal");

            if(interval < 10 * this->originalTimeFrequency)
                env->settings->updateSetting("FixedInteger.Frequency.Time", "Primal", interval);

            env->output->outputDebug(fmt::format(
                "         Iteration frequency updated to {} and time frequency updated to {} ", iters, interval));
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        env->solutionStatistics.numberOfIterationsWithoutNLPCallMIP = 0;
        env->solutionStatistics.timeLastFixedNLPCall = env->timing->getElapsedTime("Total");

        env->primalSolver->usedPrimalNLPCandidates.push_back(CAND);
    }
}

} // namespace SHOT