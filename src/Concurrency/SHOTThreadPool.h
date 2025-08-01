#pragma once

#include <cstddef>
#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>
#include <functional>
#include <future>
#include "SHOTQueue.h"

namespace SHOT
{

class IThreadTask
{
public:
    IThreadTask(void) = default;
    virtual ~IThreadTask(void) = default;
    IThreadTask(const IThreadTask& rhs) = delete;
    IThreadTask& operator=(const IThreadTask& rhs) = delete;
    IThreadTask(IThreadTask&& other) = default;
    IThreadTask& operator=(IThreadTask&& other) = default;
    virtual void execute() = 0;
};

template <typename Func> class ThreadTask : public IThreadTask
{
public:
    ThreadTask(Func&& func) : func(std::move(func)) { }
    ~ThreadTask(void) override = default;

    ThreadTask(const ThreadTask& rhs) = delete;
    ThreadTask& operator=(const ThreadTask& rhs) = delete;

    ThreadTask(ThreadTask&& other) = default;
    ThreadTask& operator=(ThreadTask&& other) = default;

    void execute() override { func(); }

private:
    Func func;
};

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
    SHOTThreadPool(const size_t numThreads) : done { false }, workQueue(), threads {}
    {

        try
        {
            for(size_t i = 0; i < numThreads; ++i)
            {
                threads.emplace_back(&SHOTThreadPool::workerThread, this);
            }
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    };

    SHOTThreadPool(const SHOTThreadPool& rhs) = delete;
    SHOTThreadPool& operator=(const SHOTThreadPool& rhs) = delete;

    SHOTThreadPool(SHOTThreadPool&& other) = delete;
    SHOTThreadPool& operator=(SHOTThreadPool&& other) = delete;

    ~SHOTThreadPool() { shutdown(); }

    template <typename Func, typename... Args> auto submitTask(Func&& func, Args&&... args)
    {
        auto boundTask = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);

        using ResultType = std::invoke_result_t<decltype(boundTask)>;

        using PackagedTask = std::packaged_task<ResultType()>;

        using TaskType = ThreadTask<PackagedTask>;

        PackagedTask task(std::move(boundTask));

        SHOTFuture<ResultType> result(task.get_future());

        workQueue.push(std::make_unique<TaskType>(std::move(task)));

        return result;
    }

private:
    void workerThread()
    {

        while(!done)
        {

            std::unique_ptr<IThreadTask> pTask;

            if(workQueue.tryPop(pTask))
            {

                pTask->execute();
            }
            else
            {
                std::this_thread::yield();
            }
        }
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

    std::atomic<bool> done;
    SHOTQueue<std::unique_ptr<IThreadTask>> workQueue;
    std::vector<std::thread> threads;
};
using ThreadPoolPtr = std::shared_ptr<SHOTThreadPool>;



};
