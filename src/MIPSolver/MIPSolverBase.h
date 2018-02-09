#pragma once
#include "vector"
#include "OSInstance.h"
#include "IMIPSolver.h"
#include "SHOTSettings.h"
#include "../ProcessInfo.h"

class MIPSolverBase
{
	private:

		std::vector<int> fixedVariableIndexes;
		vector<pair<double, double> > fixedVariableOriginalBounds;

	protected:

		std::vector<GeneratedHyperplane> generatedHyperplanes;

	public:
		MIPSolverBase();
		~MIPSolverBase();

		virtual void createHyperplane(Hyperplane hyperplane);
		virtual void createIntegerCut(std::vector<int> binaryIndexes);

		virtual void createInteriorHyperplane(Hyperplane hyperplane);

		boost::optional<std::pair<std::vector<IndexValuePair>, double>> createHyperplaneTerms(Hyperplane hyperplane);

		virtual bool getDiscreteVariableStatus();
		virtual void populateSolutionPool() = 0;
		virtual std::vector<SolutionPoint> getAllVariableSolutions();
		virtual int getNumberOfSolutions() = 0;
		virtual std::vector<double> getVariableSolution(int i) = 0;
		virtual double getObjectiveValue(int i) = 0;
		virtual double getObjectiveValue();

		virtual void presolveAndUpdateBounds();
		virtual std::pair<std::vector<double>, std::vector<double>> presolveAndGetNewBounds() = 0;

		virtual std::vector<GeneratedHyperplane>* getGeneratedHyperplanes();

		virtual pair<double, double> getCurrentVariableBounds(int varIndex) = 0;

		virtual void fixVariable(int varIndex, double value) = 0;
		virtual void fixVariables(std::vector<int> variableIndexes, std::vector<double> variableValues);
		virtual void unfixVariables();
		virtual void updateVariableBound(int varIndex, double lowerBound, double upperBound) = 0;

		virtual void updateNonlinearObjectiveFromPrimalDualBounds();

		virtual int addLinearConstraint(std::vector<IndexValuePair> elements, double constant) = 0;
		virtual int addLinearConstraint(std::vector<IndexValuePair> elements, double constant, bool isGreaterThan) = 0;

		virtual void activateDiscreteVariables(bool activate) = 0;

		bool discreteVariablesActivated;
		bool cachedSolutionHasChanged;

		bool isVariablesFixed;

		std::vector<SolutionPoint> lastSolutions;

		virtual void startTimer();
		virtual void stopTimer();

		OptProblem *originalProblem;

};
