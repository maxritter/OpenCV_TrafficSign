#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <queue>
#include <iostream>

#include "safe_queue.h"
#include "helper.h"
#include "detection.h"
#include "classification.h"
#include "video_recorder.h"

using namespace cv;
using namespace dnn;

#define PROGRAM_VERSION ("1.0")

/* Init static variables */
Net detection::detection_net_;
Net classification::classification_net_;
std::vector<String> detection::out_names_;
safe_queue<Mat> helper::frames_queue;
safe_queue<Mat> helper::processed_frames_queue;
safe_queue<std::vector<Mat> > helper::predictions_queue;
volatile sig_atomic_t helper::stop;
bool helper::use_video = false;
bool helper::record_video = false;
int helper::record_fps = 0;
std::vector<int> helper::sign_speeds = { 20, 30, 50, 60, 70, 80, 0, 100, 120 };

void run_program(VideoCapture cap, std::unique_ptr<video_recorder>& recorder)
{
	Mat frame, sign_img(helper::sign_res_display_width, helper::sign_res_display_width, CV_8UC3, Scalar::all(255));
	std::string input_fps_str = "Video: 0 FPS", detection_fps_str = "Inference: 0 FPS", limit_str = "Limit: - km/h";
	std::cout << "Program started.." << std::endl;

	namedWindow("German Traffic Sign AI", WINDOW_NORMAL);
	setWindowProperty("German Traffic Sign AI", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
	while (true)
	{
		/* Grab frame */
		if (!helper::grab_frame(cap, frame))
		{
			break;
		}

		/* Show results */
		if (!helper::predictions_queue.empty())
		{
			/* Check if a sign was detected sucessfully */
			std::vector<Mat> outs = helper::predictions_queue.get();
			Mat pred_frame = helper::processed_frames_queue.get();
			Mat sign_classify;
			detection::find_sign(pred_frame, outs, sign_classify);

			/* Do the classification */
			if (!sign_classify.empty())
			{
				classification::classify_sign(sign_classify, sign_img, limit_str);
			}

			detection_fps_str = format("Inference: %d FPS", int(helper::predictions_queue.getFPS()));
		}

		/* Draw the UI and show it on the display */
		Rect roi_sign(Point(0, frame.size().height - helper::sign_res_display_width), sign_img.size());
		sign_img.copyTo(frame(roi_sign));
		int x_add = int((helper::window_width - frame.size().width) / 2);
		Rect roi_frame(Point(x_add, 0), frame.size());
		Mat display_frame(int(helper::window_height), int(helper::window_width), CV_8UC3, Scalar::all(255));
		frame.copyTo(display_frame(roi_frame));
		input_fps_str = format("Video: %d FPS", int(helper::frames_queue.getFPS()));
		putText(display_frame, limit_str, Point(x_add, frame.size().height + 25), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(0));
		putText(display_frame, input_fps_str, Point(x_add + 220, frame.size().height + 25), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(0));
		putText(display_frame, detection_fps_str, Point(x_add + 410, frame.size().height + 25), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar::all(0));
		imshow("German Traffic Sign AI", display_frame);

		/* Send frame to video recorder */
		if (helper::record_video)
		{
			recorder->safe_mat_queue.push(display_frame);
		}

		/* User pressed key, exit */
		if (waitKey(1) > 0)
		{
			std::cout << "User key pressed, exit!" << std::endl;
			break;
		}
	}
}

int main(int ac, char** av)
{
	std::unique_ptr<video_recorder> recorder;

	/* Parse command line options */
	std::cout << "*** OpenCV Traffic Sign Recognition v" << PROGRAM_VERSION << " ***" << std::endl;
	po::variables_map vm;
	helper::parse_command_line_options(vm, ac, av);

	/* Load our models */
	if (!detection::load_model() || !classification::load_model())
	{
		std::cerr << "Error loading required models, exit.." << std::endl;
		exit(EXIT_FAILURE);
	}

	/* Open our input */
	VideoCapture cap;
	bool success;
	if (helper::use_video)
	{
		std::string video_path = vm["video"].as<std::string>();
		std::cout << "Trying to open video file: " << video_path << ".." << std::endl;
		if (!helper::is_file_exist(video_path.c_str()))
		{
			std::cerr << "Input video does not exist, exit!" << std::endl;
			exit(EXIT_FAILURE);
		}
		success = helper::open_input_source(cap, video_path);
	}
	else
	{
		std::cout << "Trying to open camera.." << std::endl;
		success = helper::open_input_source(cap);
	}
	if (!success)
	{
		exit(EXIT_FAILURE);;
	}

	/* Start all threads */
	helper::stop = false;
	std::vector<std::thread> threads;
	if (helper::record_video)
	{
		recorder = std::make_unique<video_recorder>("./record/");
		threads.emplace_back(std::thread(&video_recorder::run, recorder.get()));
	}
	threads.emplace_back(detection::processing_thread);

	/* Run our main program */
	run_program(cap, recorder);

	/* Cleanup */
	helper::stop = true;
	for (auto& th : threads)
	{
		th.join();
	}
	destroyAllWindows();
	exit(EXIT_SUCCESS);
}
