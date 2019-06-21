#pragma once

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "helper.h"

/* Define the relative path of the model here */
constexpr auto classification_model_path = ("./models/classifier.pb");

using namespace cv;
using namespace dnn;

class classification
{
	static Net classification_net_;

public:
	static bool load_model();
	static void classify_sign(const Mat& sign_classify, Mat& sign_img, std::string& limit_str);
};

inline bool classification::load_model()
{
	if (!helper::is_file_exist(classification_model_path))
	{
		return false;
	}

	classification_net_ = readNetFromTensorflow(classification_model_path, "");
	classification_net_.setPreferableBackend(DNN_BACKEND_DEFAULT);
	classification_net_.setPreferableTarget(DNN_TARGET_CPU);

	return true;
}

inline void classification::classify_sign(const Mat& sign_classify, Mat& sign_img, std::string& limit_str)
{
	/* Downsize image to helper::sign_res_input_px and make it grayscale */
	Mat blob;
	cvtColor(sign_classify, blob, COLOR_BGR2GRAY);
	resize(blob, blob, Size(helper::sign_res_input_px, helper::sign_res_input_px));
	blobFromImage(blob, blob, helper::sign_scale_input, Size(helper::sign_res_input_px, helper::sign_res_input_px), 
		Scalar(), false, false);

	classification_net_.setInput(blob);
	Mat prob = classification_net_.forward();

	Point class_id_point;
	double confidence;
	minMaxLoc(prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
	int class_id = class_id_point.x;

	if ((confidence >= helper::sign_threshold) && (class_id >= 0) && (class_id < 9))
	{
		limit_str = helper::get_limit_string(class_id);
		sign_img = sign_classify;
	}
}
