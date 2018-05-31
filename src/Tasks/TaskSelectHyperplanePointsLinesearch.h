/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "TaskBase.h"
#include "../ProcessInfo.h"
#include "../OptProblems/OptProblemOriginal.h"
#include "TaskSelectHyperplanePointsSolution.h"

class TaskSelectHyperplanePointsLinesearch : public TaskBase
{
  public:
	TaskSelectHyperplanePointsLinesearch();
	virtual ~TaskSelectHyperplanePointsLinesearch();

	virtual void run();
	virtual void run(vector<SolutionPoint> solPoints);

	virtual std::string getType();

  private:
	TaskSelectHyperplanePointsSolution *tSelectHPPts;
	bool hyperplaneSolutionPointStrategyInitialized = false;
};
