/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE
   This software is licensed under the Eclipse Public License 2.0.
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "TaskBase.h"

#include "../Structs.h"

#include <vector>

namespace SHOT
{
class TaskSelectHyperplanePointsByObjectiveRootsearch : public TaskBase
{
public:
    TaskSelectHyperplanePointsByObjectiveRootsearch(EnvironmentPtr envPtr);
    virtual ~TaskSelectHyperplanePointsByObjectiveRootsearch();

    virtual void run();
    virtual void run(std::vector<SolutionPoint> solPoints);
    virtual std::string getType();
};
} // namespace SHOT