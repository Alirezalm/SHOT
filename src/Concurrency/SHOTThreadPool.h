#pragma once

#include <algorithm>
#include <cstddef>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <future>
#include "SHOTQueue.h"

namespace SHOT
{
template <typename T> class SHOTFuture
{
public:
    SHOTFuture(std::future<T>&& future) : future(std::move(future)) { };

    SHOTFuture(const SHOTFuture& rhs) = delete;
    SHOTFuture& operator=(const SHOTFuture& rhs) = delete;

    SHOTFuture(SHOTFuture&& other) = default;
    SHOTFuture& operator=(SHOTFuture&& other) = default;

    ~SHOTFuture()
    {
        if(future.valid())
        {
            future.get();
        }
    }

    auto get() { return future.get(); }

private:
    std::future<T> future;
};

class SHOTThreadPool
{
public:
    SHOTThreadPool(const size_t numThreads) : done { false }, workQueue {}, threads {}, threadCount { numThreads } { };

    SHOTThreadPool(const SHOTThreadPool& rhs) = delete;
    SHOTThreadPool& operator=(const SHOTThreadPool& rhs) = delete;

    SHOTThreadPool(SHOTThreadPool&& other) = delete;
    SHOTThreadPool& operator=(SHOTThreadPool&& other) = delete;

    ~SHOTThreadPool() { shutdown(); }

    size_t getThreadCount() const { return threadCount; }

    void start()
    {
        try
        {
            for(size_t i = 0; i < threadCount; ++i)
            {
                threads.emplace_back(&SHOTThreadPool::workerThread, this);
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    template <typename Func> auto submitTask(Func&& func)
    {

        std::packaged_task<void()> task(std::move(func));

        SHOTFuture<void> result(task.get_future());

        workQueue.push(std::move(task));

        return result;
    }
    void shutdown()
    {
        done = true;
        workQueue.shutdown();
        for(auto& thread : threads)
        {
            if(thread.joinable())
            {
                thread.join();
            }
        }
        threads.clear();
    }

private:
    void workerThread()
    {

        while(!done)
        {

            std::packaged_task<void()> task;

            if(workQueue.tryPop(task))
            {

                task();
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }

    std::atomic<bool> done;
    SHOTQueue<std::packaged_task<void()>> workQueue;
    std::vector<std::thread> threads;
    size_t threadCount;
};
using ThreadPoolPtr = std::shared_ptr<SHOTThreadPool>;

};
