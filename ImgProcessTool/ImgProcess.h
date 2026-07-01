#pragma once
#include <QImage>
#include <QVector>
#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/opencv.hpp"   
#include <opencv2/core/core.hpp>  

class ImgProcess
{
public:
	ImgProcess();

	~ImgProcess();

	void Push(QImage& img);

	QImage cvMat2QImage(const cv::Mat& mat);

	cv::Mat QImage2cvMat(QImage img);

	QImage splitBGR(QImage src, int color);

	QImage splitColor(QImage src, QString model, int color);

	//几何变换
	//放大
	QImage enlarge(QImage src, int times);
	//旋转
	QImage rodateByCenter(QImage src, int angle);

	QImage rotate(QImage src, int angle);

	QImage flip(QImage src, int flip);

	QImage bin(QImage src, int threshold);

	QImage gray(QImage src);

	QImage Reverse(QImage src);

	QImage LogTrans(QImage src, int c);

	QImage Gamma(QImage src, int gamma);

	QImage Histeq(QImage src);

	QImage Linear(QImage src, int alpha, int beta);


	//图像增强
	QImage CircleDetect(QImage src, int minRad, int maxRad);

	QImage LineDetect(QImage src);

	QImage Normalize(QImage src, int kernelsize);

	QImage Gaussian(QImage src, int kernelsize);

	QImage Median(QImage src, int kernelsize);

	QImage Sobel(QImage src, int kernelsize);

	QImage Laplacian(QImage src, int kernelsize);

	QImage Canny(QImage src, int kernelsize, int lowthreshold, int highthreshold);

	QImage Erode(QImage src, int elem, int kernel, int times);

	QImage Dilate(QImage src, int elem, int kernel, int times);

	QImage OpenOperation(QImage src, int elem, int kernel, int times);

	QImage CloseOperation(QImage src, int elem, int kernel, int times);

	QImage TopHat(QImage src, int elem, int kernel);

	QImage BlackHat(QImage src, int elem, int kernel);

	QImage MorphologyGradient(QImage src, int elem, int kernel);

public:
	QImage img_;
};

