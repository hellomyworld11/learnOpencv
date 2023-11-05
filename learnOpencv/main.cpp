#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#include "ImgConvert.h"
#include "Opencv_mxyBook.h"
#include "LearnOpencv.h"
#include "LearnOpencv3.h"
#include "question100.h"

#define _MyMain 1
#if _MyMain
int main(int argc, char *argv[])
{
	//Mat src = imread("D:/opencvMdl/opencv_tutorial_data-master/images/wm.jpg");
	//VideoCapture vsrc("D:/opencvMdl/3D批量布线.wmv");
	//VideoCapture vsrcCama(0);
	//namedWindow("testwindow", WINDOW_AUTOSIZE);
	//imshow("input", src);
	//CImgConvert convert;
	//convert.mat_creation_demo(src);
	//CLearnOpencv lp;
	//LearnOpencv3 lp;
	//lp.smoothing();

	enum class TestMode
	{
		UAUSL,
		MXY_TEST,
		LEARN_OPENCV3,
		LEARN_OPENCV,
		QUE100
	}; 
	TestMode mode = TestMode::UAUSL;
	switch (mode)
	{
	case TestMode::UAUSL:
	{
		CImgConvert test;
		test.testMain();
		break;
	}
	case TestMode::MXY_TEST:
	{
		Opencv_mxyBook booktest;
		booktest.testMain();
		break;
	}
	case TestMode::LEARN_OPENCV3:
		break;
	case TestMode::LEARN_OPENCV:
		break;
	case TestMode::QUE100:
	{
		question100 que;
		que.testMain();
		break;
	}
	default:
		break;
	}
	
	destroyAllWindows();
	return 0;
}


#endif