#pragma once

#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "helper.h"

/* Define the relative path of the model here */
#define DETECTION_MODEL_PATH ("./models/detection.pb")
#define DETECTION_CONFIG_PATH ("./models/detection.pbtxt")

using namespace cv;
using namespace dnn;

class detection
{
	static Net detection_net_;
	static std::vector<String> out_names_;

public:
	static bool load_model();
	static bool find_sign(Mat& frame, const std::vector<Mat>& outs, Mat& sign_img);
	static void processing_thread();
};

inline bool detection::load_model()
{
	if(!helper::is_file_exist(DETECTION_MODEL_PATH) || !helper::is_file_exist(DETECTION_CONFIG_PATH))
	{
		return false;
	}

	detection_net_ = readNetFromTensorflow(DETECTION_MODEL_PATH, DETECTION_CONFIG_PATH);
	detection_net_.setPreferableBackend(DNN_BACKEND_DEFAULT);
	detection_net_.setPreferableTarget(DNN_TARGET_CPU);
	out_names_ = detection_net_.getUnconnectedOutLayersNames();

	return true;
}

inline bool detection::find_sign(Mat& frame, const std::vector<Mat>& outs, Mat& sign_img)
{
	for (const auto& out : outs)
	{
		const auto data = reinterpret_cast<float*>(out.data);
		for (size_t i = 0; i < out.total(); i += 7)
		{
			const float confidence = data[i + 2];
			const int class_id = static_cast<int>(data[i + 1]) - 1;

			/* Check if threshold and target class matches */
			if ((confidence > helper::sign_threshold) && (class_id == helper::target_class_id))
			{
				const float left = data[i + 3];
				const float top = data[i + 4];
				const float right = data[i + 5];
				const float bottom = data[i + 6];

				const int im_width = frame.size().width;
				const int im_height = frame.size().height;

				const auto box_left = static_cast<int>(std::max(int((left - helper::sign_extend_px) * im_width), 0));
				const auto box_top = static_cast<int>(std::max(int((top - helper::sign_extend_px) * im_height), 0));
				const auto box_right = static_cast<int>(std::min(int((right + helper::sign_extend_px) * im_width), im_width - 1));
				const auto box_bottom = static_cast<int>(std::min(int((bottom + helper::sign_extend_px) * im_height), im_height - 1));

				/* Cut out the sign and resize it */
				sign_img = frame(Rect(box_left, box_top, abs(box_right - box_left), abs(box_bottom - box_top))).clone();
				resize(sign_img, sign_img, Size(helper::sign_res_display_width, helper::sign_res_display_width), INTER_CUBIC);
				return true;
			}
		}
	}

	return false;
}

inline void detection::processing_thread()
{
	Mat blob;
	std::cout << "Processing Thread started.." << std::endl;
	while (!helper::stop)
	{
		/* Get a next frame */
		Mat frame;
		{
			if (!helper::frames_queue.empty())
			{
				frame = helper::frames_queue.get();
				/* Skip the rest of frames */
				helper::frames_queue.clear();
			}
		}

		/* Run the inference */
		if (!frame.empty())
		{
			blobFromImage(frame, blob, 1.0, Size(helper::detection_input_width, 
				helper::detection_input_height), Scalar(), true);
			detection_net_.setInput(blob);

			std::vector<Mat> outs;
			detection_net_.forward(outs, out_names_);
			helper::predictions_queue.push(outs);
			helper::processed_frames_queue.push(frame);
		}
	}
}
