/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "TaskBase.h"

#include <memory>
#include <string>
#include <vector>

#include "../Structs.h"
#include <mutex>

namespace SHOT
{
class INLPSolver;

using NLPSolverPtr = std::shared_ptr<INLPSolver>;

class TaskParallelSelectPrimalCandidatesFromNLP : public TaskBase
{
public:
    TaskParallelSelectPrimalCandidatesFromNLP(EnvironmentPtr envPtr, bool useReformulatedProblem);
    ~TaskParallelSelectPrimalCandidatesFromNLP() override;
    void run() override;
    std::string getType() override;

private:

    bool parallelSolveFixedNLP();
    void processCandidate(PrimalFixedNLPCandidate CAND);
    NLPSolverPtr createNLPSolver(bool useReformulatedProblem); //runs in serial
    void createInfeasibilityCut(const VectorDouble point);
    void createIntegerCut(VectorDouble point);

    // std::shared_ptr<INLPSolver> NLPSolver;

    // VectorInteger discreteVariableIndexes;
    std::vector<VectorDouble> testedPoints;
    VectorDouble fixPoint;

    double originalNLPTime;
    double originalNLPIter;

    VectorDouble originalLBs;
    VectorDouble originalUBs;

    // VectorString variableNames;

    std::shared_ptr<TaskBase> taskSelectHPPts;

    int originalIterFrequency;
    double originalTimeFrequency;

    // ProblemPtr sourceProblem;
    bool sourceIsReformulatedProblem = false;
    bool useReformulatedProblem = false;
    std::mutex mutex;
};
} // namespace SHOT