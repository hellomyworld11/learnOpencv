#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class CImgConvert
{
public:
	void colorSpace_demo(Mat &image);
	void mat_creation_demo(Mat &image);
	void pixel_visit_demo(Mat &image);
	void operators_demo(Mat &image);
	void tracking_bar_demo(Mat &image);
	void color_style_demo(Mat &image);
	void bitwise_demo(Mat &image);
};

