#pragma once
#include "SHOTThreadPool.h"
#include <algorithm>
#include <vector>

namespace SHOT
{

class ParallelSHOT
{
public:
    ParallelSHOT(size_t numThreads = std::thread::hardware_concurrency())
    {
        if(numThreads < 1)
        {
            throw std::invalid_argument("Number of threads must be at least 1.");
        }

        auto numHardwareThreads = static_cast<size_t>(std::thread::hardware_concurrency());

        if(numThreads > numHardwareThreads)
        {
            numThreads = static_cast<size_t>(std::min(numThreads, numHardwareThreads));
        }

        threadPool = std::make_shared<SHOTThreadPool>(numThreads);
    };

    ~ParallelSHOT() { stopThreadPool(); };

    void startThreadPool()
    {
        if(threadPool)
        {
            threadPool->start();
        }
    }

    void stopThreadPool()
    {
        if(threadPool)
        {
            threadPool->shutdown();
        }
    }

    ThreadPoolPtr getThreadPool() const { return threadPool; }

    size_t getThreadCount() const
    {
        if(threadPool)
        {
            return threadPool->getThreadCount();
        }
        return 0;
    }

    void submitTask(std::function<void()>&& task)
    {
        if(!threadPool)
        {
            throw std::runtime_error("Thread pool is not initialized.");
        }

        auto future = threadPool->submitTask(std::move(task));
        
        futures.push_back(std::move(future));
    }

    void waitForAllTasks()
    {
        for(auto& fut : futures)
        {
            fut.get();
        }
        
        futures.clear();
    }

    void initializeAutoDiff() {
        // todo
    };

private:
    ThreadPoolPtr threadPool;

    std::vector<SHOTFuture<void>> futures;
};

using ParallelSHOTPtr = std::shared_ptr<ParallelSHOT>;

}