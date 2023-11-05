#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

//参考官方文档 https://docs.opencv.org/4.x/de/d7a/tutorial_table_of_content_core.html
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
	//预处理
	Mat preProcess(Mat img);

	//官方
	//3.2 算术运算
	//加法
	void add_test();
	//图像融合
	void addWeighted_test();
	//按位运算
	void addLogo();
	//4.1 颜色空间
	void colorspace();
	//4.2 变换
	//缩放
	void resize_test();
	//平移
	void move();
	//旋转
	void rotate();
	//仿射
	void Affine();
	//透视
	void warp();
	//4.3 图像阈值
	//4.4 图像平滑
	//2D 卷积
	void filter2D_test();
	//均值滤波
	void blur_test();
	//高斯模糊
	void GaussianBlur_test();
	//4.5 形态学转换
	//腐蚀
	void erode_test();
	//膨胀
	void dilate_test();
	//开运算  去除白色噪点
	void open();
	//闭运算  去除白色区域内部的黑色噪点
	void close();
	//形态学梯度 获取轮廓
	void graint_test();
	//顶帽 输入图像和图像开运算之差
	//黑帽 输入图像和图像闭运算之差
	//4.6 图像梯度
	//Sobel Laplacian
	void Sobel_laplacian_test();
	//4.7 边缘检测
	void canny_test();
	//4.8 图像金字塔 ？
	void pyramid_test();
	//4.9.1 轮廓
	void contour_test();
	//4.9.2 轮廓特征
	void contour_feattest();
	//凸包
	void convexhull();
	//bounding box
	void boundingbox();
	//4.10.1 直方图
	void hist_test();
	//4.11 傅里叶变换 ?
	void discrete_fourier();
	//4.12 模板匹配
	void match_template();
	//4.13 霍夫线变换 HoughLines
	//4.15 图像分割 分水岭算法
	void watershed_test();
};

