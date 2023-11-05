#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <random>
using namespace cv;
// 计算机编程视觉攻略 


class CLearnOpencv
{
public:
	CLearnOpencv();
	~CLearnOpencv();
	//图片加载 显示 保存
	void mat_load_show_save();
	//视频加载
	void video_Capture();
	void camera_capture();
	//在图像上绘图
	void mat_drw_pic();
	//mat 
	void mat_test();
	//ROI
	void mat_ROI();
	//噪声
	void salt_test();
	//减色
	void reduce_test();
	//锐色
	void sharpen_test();
	//图像相加
	void pic_add();
	//波浪
	void wave_test();
	//GrabCut
	void grabcut_test();
	//直方图
	void hispic_test();
private:
	void salt(cv::Mat image, int n);
	void reduce(cv::Mat image, int div = 64);
	void sharpen(const cv::Mat& image, cv::Mat &result);
	void wave(const cv::Mat& image, cv::Mat& result);
	void grabcut(const cv::Mat& image, cv::Mat& result);
	void calchist(const cv::Mat& image, cv::Mat &result);

};

