/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "../Shared.h"
#include "IMIPSolver.h"

#include "IRelaxationStrategy.h"
#include "RelaxationStrategyStandard.h"
#include "RelaxationStrategyNone.h"

namespace SHOT
{
class MIPSolverBase
{
private:
    VectorInteger fixedVariableIndexes;
    std::vector<PairDouble> fixedVariableOriginalBounds;

    bool auxilliaryObjectiveVariableDefined = false;
    int auxilliaryObjectiveVariableIndex;

protected:
    std::vector<GeneratedHyperplane> generatedHyperplanes;

    int numberOfVariables = 0;
    int numberOfConstraints = 0;
    bool isMinimizationProblem;
    bool isProblemDiscrete = false;
    std::vector<E_VariableType> variableTypes;
    VectorDouble variableLowerBounds;
    VectorDouble variableUpperBounds;
    VectorString variableNames;

    bool cutOffConstraintDefined = false;
    int cutOffConstraintIndex;

public:
    ~MIPSolverBase();

    virtual void createHyperplane(Hyperplane hyperplane);
    // virtual void createIntegerCut(VectorInteger &binaryIndexes);

    virtual void createInteriorHyperplane(Hyperplane hyperplane);

    std::optional<std::pair<std::vector<PairIndexValue>, double>> createHyperplaneTerms(Hyperplane hyperplane);

    virtual void setCutOffAsConstraint(double cutOff) = 0;

    virtual bool getDiscreteVariableStatus();

    virtual E_IterationProblemType getCurrentProblemType();
    virtual void executeRelaxationStrategy();

    virtual std::vector<SolutionPoint> getAllVariableSolutions();
    virtual int getNumberOfSolutions() = 0;
    virtual VectorDouble getVariableSolution(int i) = 0;
    virtual double getObjectiveValue(int i) = 0;
    virtual double getObjectiveValue();

    virtual void presolveAndUpdateBounds();
    virtual std::pair<VectorDouble, VectorDouble> presolveAndGetNewBounds() = 0;

    virtual std::vector<GeneratedHyperplane>* getGeneratedHyperplanes();

    virtual PairDouble getCurrentVariableBounds(int varIndex) = 0;

    virtual void fixVariable(int varIndex, double value) = 0;
    virtual void fixVariables(VectorInteger variableIndexes, VectorDouble variableValues);
    virtual void unfixVariables();
    virtual void updateVariableBound(int varIndex, double lowerBound, double upperBound) = 0;

    virtual int addLinearConstraint(const std::vector<PairIndexValue>& elements, double constant, std::string name) = 0;
    virtual int addLinearConstraint(
        const std::vector<PairIndexValue>& elements, double constant, std::string name, bool isGreaterThan)
        = 0;

    virtual void activateDiscreteVariables(bool activate) = 0;

    virtual int getNumberOfExploredNodes() = 0;
    virtual int getNumberOfOpenNodes();

    virtual bool hasAuxilliaryObjectiveVariable() { return auxilliaryObjectiveVariableDefined; };
    virtual int getAuxilliaryObjectiveVariableIndex() { return auxilliaryObjectiveVariableIndex; };

    virtual void setAuxilliaryObjectiveVariableIndex(int index)
    {
        auxilliaryObjectiveVariableIndex = index;
        auxilliaryObjectiveVariableDefined = true;
    };

    int prevSolutionLimit = 1;

    bool discreteVariablesActivated;
    bool cachedSolutionHasChanged;
    bool modelUpdated = true;

    bool isVariablesFixed = false;
    bool alreadyInitialized = false;

    std::vector<SolutionPoint> lastSolutions;

    std::unique_ptr<IRelaxationStrategy> relaxationStrategy;

    EnvironmentPtr env;
};
} // namespace SHOT