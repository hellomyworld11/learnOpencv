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
	void test_img_drwGeom(Mat &src);  //�û��Ƽ��η�ʽ���ƻ�

	void Roi_Addimage();
	void Linear_Blending();
	void pic_ligth_contrast_control();


	void test_morphology();

	//��Ե���
	void test_canny();

	//ֱ��ͼ���⻯
	void test_equalizehist();

	//��������
	void contour_find();

	//��̬�������
	void contour_find_dynamic();

	//͹��
	void converxhull();

	//��ˮ���㷨


	//ͼ���޲�
	void picinpaint();

	//�ǵ���
	//1.harris
	void harris_dec();

private:
	void DrawEllipse(Mat img, double angle);
	void DrawFilledCircle(Mat img, Point center);
	void DrawLine(Mat img, Point start, Point end);
};

