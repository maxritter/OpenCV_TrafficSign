#pragma once

#include <queue>
#include <mutex>
#include <opencv2/core/utility.hpp>

template <typename T> class safe_queue : public std::queue<T>
{
public:
	safe_queue() : counter(0) {}

	void push(const T& entry)
	{
		std::lock_guard<std::mutex> lock(mutex_);

		std::queue<T>::push(entry);
		counter += 1;

		/* Start counting from a second frame (warmup) */
		if (counter == 1)
		{
			tm_.reset();
			tm_.start();
		}
	}

	T get()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		T entry = this->front();
		this->pop();
		return entry;
	}

	float getFPS()
	{
		tm_.stop();
		const double fps = counter / tm_.getTimeSec();
		tm_.start();
		return static_cast<float>(fps);
	}

	void clear()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		while (!this->empty())
		{
			this->pop();
		}
	}

	unsigned int counter;

private:
	cv::TickMeter tm_;
	std::mutex mutex_;
};
