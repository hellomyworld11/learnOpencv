#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <random>
using namespace cv;
// ���������Ӿ����� 


class CLearnOpencv
{
public:
	CLearnOpencv();
	~CLearnOpencv();
	//ͼƬ���� ��ʾ ����
	void mat_load_show_save();
	//��Ƶ����
	void video_Capture();
	void camera_capture();
	//��ͼ���ϻ�ͼ
	void mat_drw_pic();
	//mat 
	void mat_test();
	//ROI
	void mat_ROI();
	//����
	void salt_test();
	//��ɫ
	void reduce_test();
	//��ɫ
	void sharpen_test();
	//ͼ�����
	void pic_add();
	//����
	void wave_test();
	//GrabCut
	void grabcut_test();
	//ֱ��ͼ
	void hispic_test();
private:
	void salt(cv::Mat image, int n);
	void reduce(cv::Mat image, int div = 64);
	void sharpen(const cv::Mat& image, cv::Mat &result);
	void wave(const cv::Mat& image, cv::Mat& result);
	void grabcut(const cv::Mat& image, cv::Mat& result);
	void calchist(const cv::Mat& image, cv::Mat &result);

};

