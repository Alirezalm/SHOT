/**
   The Supporting Hyperplane Optimization Toolkit (SHOT).

   @author Andreas Lundell, Åbo Akademi University

   @section LICENSE 
   This software is licensed under the Eclipse Public License 2.0. 
   Please see the README and LICENSE files for more information.
*/

#pragma once
#include "vector"
#include "list"
#include "unordered_map"
#include "algorithm"
#include "Tasks/TaskBase.h"
#include "Tasks/TaskException.h"

class TaskHandler
{
  public:
    TaskHandler();
    ~TaskHandler();

    void addTask(TaskBase *task, std::string taskID);
    bool getNextTask(TaskBase *&task);
    void setNextTask(std::string taskID);
    void clearTasks();

    TaskBase *getTask(std::string taskID);

  private:
    std::list<std::pair<std::string, TaskBase *>>::iterator nextTask;
    std::string nextTaskID;
    std::list<std::pair<std::string, TaskBase *>> taskIDMap;
};
