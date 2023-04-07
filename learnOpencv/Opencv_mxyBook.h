#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

class Opencv_mxyBook
{
public:
	Opencv_mxyBook();
	~Opencv_mxyBook();
private:
	void DrawEllipse(Mat img,double angle);
	void DrawFilledCircle(Mat img, Point center);
	void DrawLine(Mat img,Point start,Point end);
public:
	void test_img_structur(Mat &image);
	void test_img_vague(Mat &src);
	void test_avi_read(VideoCapture &src);
	void test_img_add(Mat &image);    
	void test_img_drwGeom(Mat &src);  //用绘制几何方式绘制花

};

