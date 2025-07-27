#pragma once

#include "../Tasks/TaskBase.h"

namespace SHOT
{
class TaskParallel : public TaskBase
{
public:
    virtual ~TaskParallel();
    void operator()() { run(); }
};
}

