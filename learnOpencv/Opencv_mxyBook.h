#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

class Opencv_mxyBook
{
public:
	Opencv_mxyBook();
	~Opencv_mxyBook();

	void testMain();

public:
	void camera_video();
	void test_img_structur(Mat &image);
	void test_img_vague(Mat &src);
	void test_avi_read(VideoCapture &src);
	void test_img_add(Mat &image);    
	void test_img_drwGeom(Mat &src);  //用绘制几何方式绘制花

	void Roi_Addimage();
	void Linear_Blending();
	void pic_ligth_contrast_control();


	void test_morphology();

	//边缘检测
	void test_canny();

	//直方图均衡化
	void test_equalizehist();

	//查找轮廓
	void contour_find();

	//动态检测轮廓
	void contour_find_dynamic();

	//凸包
	void converxhull();

	//分水岭算法


	//图像修补
	void picinpaint();

	//角点检测
	//1.harris
	void harris_dec();

private:
	void DrawEllipse(Mat img, double angle);
	void DrawFilledCircle(Mat img, Point center);
	void DrawLine(Mat img, Point start, Point end);
};

