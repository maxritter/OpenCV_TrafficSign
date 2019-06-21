#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <utility>

#include "helper.h"
#include "safe_queue.h"

using namespace cv;
namespace fs = boost::filesystem;
using namespace std::chrono;

class video_recorder
{
	std::string dir_;
	int codec_ = VideoWriter::fourcc('M', 'J', 'P', 'G');
	bool first_time_ = true;
	std::string time_stamp_;
	VideoWriter vw_;

	void _run(const Mat& m);

public:
	safe_queue<Mat> safe_mat_queue;
	video_recorder(std::string dir) : dir_(std::move(dir))
	{
		if (!fs::is_directory(dir_) || !fs::exists(dir_))
		{ 
			fs::create_directory(dir_);
		}
	};
	~video_recorder() = default;
	void run();	
};

inline void video_recorder::_run(const Mat& m)
{
	static auto save_timer = high_resolution_clock::now();

	if (!m.empty())
	{
		if (first_time_)
		{
			time_stamp_ = to_iso_string(boost::posix_time::second_clock::local_time());
			std::cout << "Created first video file: " << dir_ << time_stamp_ << ".avi" << std::endl;
			vw_.open(dir_ + time_stamp_ + ".avi", codec_, helper::record_fps, Size(int(helper::window_width), int(helper::window_height)), true);
			first_time_ = false;
		}

		Mat frame_out;
		resize(m, frame_out, Size(int(helper::window_width), int(helper::window_height)));
		vw_.write(frame_out);

		if (duration_cast<seconds>(high_resolution_clock::now() - save_timer).count() >= helper::record_save_time_s)
		{
			time_stamp_ = to_iso_string(boost::posix_time::second_clock::local_time());
			std::cout << "Saving video to file: " << dir_ << time_stamp_ << ".avi" << std::endl;
			vw_.open(dir_ + time_stamp_ + ".avi", codec_, helper::record_fps, Size(int(helper::window_width), int(helper::window_height)), true);

			save_timer = high_resolution_clock::now();
		}
	}
}

inline void video_recorder::run()
{
	try
	{
		while (!helper::stop)
		{
			if(!safe_mat_queue.empty())
			{
				Mat t = safe_mat_queue.get();
				_run(t);
			}
		}
	}
	catch (...)
	{
	}
}
