/*
 * TaskSelectPrimalCandidatesFromLinesearch.cpp
 *
 *  Created on: Apr 7, 2015
 *      Author: alundell
 */

#include "TaskUpdateNonlinearObjectiveByLinesearch.h"

TaskUpdateNonlinearObjectiveByLinesearch::TaskUpdateNonlinearObjectiveByLinesearch()
{

}

TaskUpdateNonlinearObjectiveByLinesearch::~TaskUpdateNonlinearObjectiveByLinesearch()
{
	// TODO Auto-generated destructor stub
}

void TaskUpdateNonlinearObjectiveByLinesearch::run()
{
	ProcessInfo::getInstance().startTimer("ObjectiveLinesearch");

	ProcessInfo::getInstance().setObjectiveUpdatedByLinesearch(false);

	auto currIter = ProcessInfo::getInstance().getCurrentIteration();

	if (!currIter->isMILP()) return;

	auto allSolutions = currIter->solutionPoints;

	for (int i = 0; i < allSolutions.size(); i++)
	{
		if (allSolutions.at(i).maxDeviation.value < 0) continue;

		auto oldObjVal = allSolutions.at(i).objectiveValue;
		auto changed = updateObjectiveInPoint(allSolutions.at(i));

		// Update the iteration solution as well (for i==0)
		if (changed && i == 0 /*&& (currIter->solutionStatus == E_ProblemSolutionStatus::Optimal*/)
		{
			currIter->maxDeviation = allSolutions.at(i).maxDeviation.value;
			currIter->maxDeviationConstraint = allSolutions.at(i).maxDeviation.idx;
			currIter->objectiveValue = allSolutions.at(0).objectiveValue;

			ProcessInfo::getInstance().setObjectiveUpdatedByLinesearch(true);

			auto diffobj = abs(oldObjVal - allSolutions.at(i).objectiveValue);

			// Change the status of the solution if it has been updated much
			if (diffobj > Settings::getInstance().getDoubleSetting("GapTermTolAbsolute", "Algorithm"))
			{
				if (currIter->solutionStatus == E_ProblemSolutionStatus::Optimal)
				{
					currIter->solutionStatus = E_ProblemSolutionStatus::SolutionLimit;
				}
			}
		}
	}

	ProcessInfo::getInstance().stopTimer("ObjectiveLinesearch");
}

std::string TaskUpdateNonlinearObjectiveByLinesearch::getType()
{
	std::string type = typeid(this).name();
	return (type);

}

bool TaskUpdateNonlinearObjectiveByLinesearch::updateObjectiveInPoint(SolutionPoint &solution)
{
	std::vector<int> constrIdxs(1);
	constrIdxs.push_back(-1);

	auto dualSol = solution;

	auto oldObjVal = solution.objectiveValue;

	double mu = dualSol.objectiveValue;
	double error = ProcessInfo::getInstance().originalProblem->calculateConstraintFunctionValue(-1, dualSol.point);

	vector<double> tmpPoint(dualSol.point);
	tmpPoint.back() = mu + 1.05 * error;

	std::vector<double> internalPoint;
	std::vector<double> externalPoint;

	bool changed = false;

	try
	{
		auto xNewc = ProcessInfo::getInstance().linesearchMethod->findZero(tmpPoint, dualSol.point,
				Settings::getInstance().getIntSetting("LinesearchMaxIter", "Linesearch"),
				Settings::getInstance().getDoubleSetting("LinesearchLambdaEps", "Linesearch"), 0, constrIdxs);

		internalPoint = xNewc.first;
		externalPoint = xNewc.second;

		auto mostDevInner = ProcessInfo::getInstance().originalProblem->getMostDeviatingConstraint(internalPoint);
		auto mostDevOuter = ProcessInfo::getInstance().originalProblem->getMostDeviatingConstraint(externalPoint);

		solution.maxDeviation = mostDevOuter;
		solution.objectiveValue = ProcessInfo::getInstance().originalProblem->calculateOriginalObjectiveValue(
				externalPoint);
		solution.point.back() = externalPoint.back();

		if (oldObjVal != solution.objectiveValue) changed = true;
		auto diffobj = abs(oldObjVal - solution.objectiveValue);

		if (changed)
		{
			if (diffobj > Settings::getInstance().getDoubleSetting("GapTermTolAbsolute", "Algorithm"))
			{
				Hyperplane hyperplane;
				hyperplane.sourceConstraintIndex = mostDevOuter.idx;
				hyperplane.generatedPoint = externalPoint;
				hyperplane.source = E_HyperplaneSource::PrimalSolutionSearch;
				ProcessInfo::getInstance().hyperplaneWaitingList.push_back(hyperplane);

				ProcessInfo::getInstance().outputInfo(
						"     Obj. for sol. # 0 upd. by l.s. " + UtilityFunctions::toString(oldObjVal) + " -> "
								+ UtilityFunctions::toString(solution.objectiveValue) + " (diff:"
								+ UtilityFunctions::toString(diffobj) + ")  #");
			}
			else
			{
				ProcessInfo::getInstance().outputInfo(
						"     Obj. for sol. # 0 upd. by l.s. " + UtilityFunctions::toString(oldObjVal) + " -> "
								+ UtilityFunctions::toString(solution.objectiveValue) + " (diff:"
								+ UtilityFunctions::toString(diffobj) + ")  ");
			}
		}

		ProcessInfo::getInstance().addPrimalSolutionCandidate(internalPoint, E_PrimalSolutionSource::Linesearch,
				ProcessInfo::getInstance().getCurrentIteration()->iterationNumber);

	}
	catch (std::exception &e)
	{
		ProcessInfo::getInstance().outputWarning(
				"     Cannot find solution with linesearch for updating nonlinear objective.");
	}

	return (changed);
}
