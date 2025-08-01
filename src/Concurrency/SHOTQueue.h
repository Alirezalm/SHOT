#pragma once

#include <iostream>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>

namespace SHOT
{
template <typename Task> class SHOTQueue
{
public:
    SHOTQueue() : stop(false) {

     }

    SHOTQueue(const SHOTQueue&) = delete;
    SHOTQueue& operator=(const SHOTQueue&) = delete;

    SHOTQueue(SHOTQueue&&) = delete;
    SHOTQueue& operator=(SHOTQueue&&) = delete;

    ~SHOTQueue() { shutdown(); };

    bool tryPop(Task& task)
    {

        std::unique_lock<std::mutex> lock(mtx);

        if(queue.empty())
        {
            return false;
        }

        task = std::move(queue.front());
        queue.pop();
        return true;
    }

    std::shared_ptr<Task> tryPop()
    {
        std::unique_lock<std::mutex> lock(mtx);

        if(queue.empty())
        {
            return std::shared_ptr<Task>();
        }

        std::shared_ptr<Task> taskPtr = std::make_shared<Task>(std::move(queue.front()));
        queue.pop();
        return taskPtr;
    }

    bool waitAndPop(Task& task)
    {
        std::unique_lock<std::mutex> lock(mtx);

        cv.wait(lock, [this]() -> bool { return !queue.empty() || stop; });

        if(queue.empty())
        {
            return false;
        }

        task = std::move(queue.front());

        queue.pop();

        return true;
    }

    std::shared_ptr<Task> waitAndPop()
    {
        std::unique_lock<std::mutex> lock(mtx);

        cv.wait(lock, [this]() -> bool { return !queue.empty() || stop; });

        if(queue.empty())
        {
            return std::shared_ptr<Task>();
        }

        std::shared_ptr<Task> taskPtr = std::make_shared<Task>(std::move(queue.front()));

        queue.pop();

        return taskPtr;
    }

    void push(Task task)
    {

        std::unique_lock<std::mutex> lock(mtx);

        if(stop)
        {
            throw std::runtime_error("Cannot push to a stopped queue");
        }

        queue.push(std::move(task));

        cv.notify_one();
    }

    bool isStopped() const { return stop; }

    void shutdown()
    {

        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }

        cv.notify_all();
    }
    bool isEmpty() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

private:
    std::queue<Task> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop;
};
}