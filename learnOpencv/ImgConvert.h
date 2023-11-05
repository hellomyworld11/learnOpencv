#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

//�ο��ٷ��ĵ� https://docs.opencv.org/4.x/de/d7a/tutorial_table_of_content_core.html
class TimeL
{
public:
	TimeL()
	{
		t1 = getTickCount();
	}
	~TimeL()
	{
		t2 = getTickCount();
		long long time = (t2 - t1) / getTickFrequency();
		std::cout << "time: " << time << std::endl;
	}
	void Print()
	{
		t2 = getTickCount();
		long long time = (t2 - t1) / getTickFrequency();
		std::cout << "time: " << time << std::endl;
		t1 = getTickCount();
	}
private:
	long long t1;
	long long t2;
};

class CImgConvert
{
public:
	void testMain();

private:
	void colorSpace_demo(Mat &image);
	void mat_creation_demo(Mat &image);
	void pixel_visit_demo(Mat &image);
	void operators_demo(Mat &image);
	void tracking_bar_demo(Mat &image);
	void color_style_demo(Mat &image);
	void bitwise_demo(Mat &image);

	//4h
	void camera();
	
	void colorDectection();
	void contourDectection();
	void getContours(Mat imgDil, Mat img);
	void shapeDectection();
	void faceDectection();
	void document_scanner();
	//Ԥ����
	Mat preProcess(Mat img);

	//�ٷ�
	//3.2 ��������
	//�ӷ�
	void add_test();
	//ͼ���ں�
	void addWeighted_test();
	//��λ����
	void addLogo();
	//4.1 ��ɫ�ռ�
	void colorspace();
	//4.2 �任
	//����
	void resize_test();
	//ƽ��
	void move();
	//��ת
	void rotate();
	//����
	void Affine();
	//͸��
	void warp();
	//4.3 ͼ����ֵ
	//4.4 ͼ��ƽ��
	//2D ���
	void filter2D_test();
	//��ֵ�˲�
	void blur_test();
	//��˹ģ��
	void GaussianBlur_test();
	//4.5 ��̬ѧת��
	//��ʴ
	void erode_test();
	//����
	void dilate_test();
	//������  ȥ����ɫ���
	void open();
	//������  ȥ����ɫ�����ڲ��ĺ�ɫ���
	void close();
	//��̬ѧ�ݶ� ��ȡ����
	void graint_test();
	//��ñ ����ͼ���ͼ������֮��
	//��ñ ����ͼ���ͼ�������֮��
	//4.6 ͼ���ݶ�
	//Sobel Laplacian
	void Sobel_laplacian_test();
	//4.7 ��Ե���
	void canny_test();
	//4.8 ͼ������� ��
	void pyramid_test();
	//4.9.1 ����
	void contour_test();
	//4.9.2 ��������
	void contour_feattest();
	//͹��
	void convexhull();
	//bounding box
	void boundingbox();
	//4.10.1 ֱ��ͼ
	void hist_test();
	//4.11 ����Ҷ�任 ?
	void discrete_fourier();
	//4.12 ģ��ƥ��
	void match_template();
	//4.13 �����߱任 HoughLines
	//4.15 ͼ��ָ� ��ˮ���㷨
	void watershed_test();
};

