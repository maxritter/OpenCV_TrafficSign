#pragma once

#include <csignal>
#include <fstream>
#include <boost/program_options.hpp>

using namespace cv;
namespace po = boost::program_options;

class helper
{
public:
	static bool use_video;
	static bool record_video;
	static const int sign_res_input_px = 32;
	static const int sign_res_display_width = 128;
	constexpr static float sign_scale_input = 0.003921f; // Normalize to 1/255
	constexpr static float sign_extend_px = 0.008f;
	constexpr static float sign_threshold = 0.99f;
	static const int target_class_id = 0;
	static const int detection_input_width = 300;
	static const int detection_input_height = 300;
	constexpr static float input_height = 440.0f;
	constexpr static float window_width = 800.0f;
	constexpr static float window_height = 480.0f;
	static const int record_save_time_s = 60;
	static int record_fps;
	static safe_queue<Mat> frames_queue;
	static safe_queue<Mat> processed_frames_queue;
	static safe_queue<std::vector<Mat> > predictions_queue;
	static volatile sig_atomic_t stop;

	static bool grab_frame(VideoCapture& cap, Mat& frame);
	static bool is_file_exist(const char* file_name);
	static bool open_input_source(VideoCapture& cap, const std::string& input_src);
	static void parse_command_line_options(po::variables_map& vm, int ac, char** av);
	static std::vector<int> sign_speeds;
	static std::string get_limit_string(int class_id);
};

inline bool helper::grab_frame(VideoCapture& cap, Mat& frame)
{
	/* Grab frame and add it to the frame queue */
	cap >> frame;
	if (!frame.empty())
	{
		helper::frames_queue.push(frame.clone());
	}
	else if (helper::use_video)
	{
		std::cout << "End of video reached, exit.." << std::endl;
		return false;
	}
	else
	{
		std::cerr << "Error connecting to camera, exit.." << std::endl;
		return false;
	}

	/* Downsize input image for display */
	const float ratio = helper::input_height / frame.size().height;
	resize(frame, frame, Size(), ratio, ratio, INTER_LINEAR);
	return true;
}

inline bool helper::is_file_exist(const char* file_name)
{
	std::ifstream infile(file_name);
	return infile.good();
}

inline bool helper::open_input_source(VideoCapture& cap, const std::string& input_src = "")
{
	try
	{
		/* Open Webcam */
		if (input_src.empty())
		{
			cap.open(0);
			record_fps = int(cap.get(CAP_PROP_FPS));
			int cam_width = int(cap.get(CAP_PROP_FRAME_WIDTH));
			int cam_height = int(cap.get(CAP_PROP_FRAME_HEIGHT));
			std::cout << "Opened camera with resolution " << cam_width << "x" << cam_height << " and FPS: " << record_fps << ".." << std::endl;
		}

		/* Open Video */
		else
		{
			cap.open(input_src);
			int video_width = int(cap.get(CAP_PROP_FRAME_WIDTH));
			int video_height = int(cap.get(CAP_PROP_FRAME_HEIGHT));
			std::cout << "Opened video with resolution " << video_width << "x" << video_height << ".." << std::endl;
		}
	}
	catch(...)
	{
		std::cerr << "Unable to open our input source, exit!" << std::endl;
		return false;
	}

	if (!cap.isOpened())
	{
		std::cerr << "Unable to open our input source, exit!" << std::endl;
		return false;
	}

	return true;
}

inline void helper::parse_command_line_options(po::variables_map& vm, int ac, char** av)
{
	try
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("video,v", po::value<std::string>(), "use test video instead of camera")
			("record,r", "record output to video file");

		store(parse_command_line(ac, av, desc), vm);
		notify(vm);

		/* Show help */
		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			exit(EXIT_FAILURE);
		}

		/* Eventually use test video instead of camera */
		if (vm.count("video"))
		{
			std::cout << "Using video instead of camera input.." << std::endl;
			use_video = true;
		}

		/* Eventually record display to video file */
		if (vm.count("record"))
		{
			std::cout << "Recording video with resolution " << window_width << "x" << window_height <<
				" and " << record_fps << " FPS.." << std::endl;
			record_video = true;
		}
	}
	catch (boost::program_options::error& e)
	{
		std::cerr << "ERROR: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
}

inline std::string helper::get_limit_string(const int class_id)
{
	const int speed = sign_speeds[class_id];
	if(speed != 0)
	{
		return "Limit: " + std::to_string(speed) + " km/h";
	}

	return "Limit: - km/h";
}
