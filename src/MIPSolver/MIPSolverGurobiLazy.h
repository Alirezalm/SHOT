/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "MIPSolverBase.h"
#include "MIPSolverGurobi.h"
#include "MIPSolverCallbackBase.h"

namespace SHOT
{

class GurobiCallback : public GRBCallback, public MIPSolverCallbackBase
{
public:
    GRBVar* vars;
    GurobiCallback(GRBVar* xvars, EnvironmentPtr envPtr);
    ~GurobiCallback();

protected:
    void callback() override;

private:
    int lastExploredNodes = 0;
    int lastOpenNodes = 0;

    bool createHyperplane(Hyperplane hyperplane);

    virtual bool createIntegerCut(VectorInteger& binaryIndexesOnes, VectorInteger& binaryIndexesZeroes);

    void addLazyConstraint(std::vector<SolutionPoint> candidatePoints);
};

class MIPSolverGurobiLazy : public MIPSolverGurobi
{
public:
    MIPSolverGurobiLazy(EnvironmentPtr envPtr);
    ~MIPSolverGurobiLazy() override;

    void checkParameters() override;

    void initializeSolverSettings() override;

    int increaseSolutionLimit(int increment) override;
    void setSolutionLimit(long limit) override;
    int getSolutionLimit() override;

    E_ProblemSolutionStatus solveProblem() override;

private:
    bool isCallbackInitialized = false;
    std::unique_ptr<GurobiCallback> gurobiCallback;
};
} // namespace SHOT