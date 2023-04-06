#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

class Opencv_mxyBook
{
public:
	Opencv_mxyBook();
	~Opencv_mxyBook();
public:
	void test_img_structur(Mat &image);
	void test_img_vague(Mat &src);
	void test_avi_read(VideoCapture &src);
};

