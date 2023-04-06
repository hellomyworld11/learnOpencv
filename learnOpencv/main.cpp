#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#include "ImgConvert.h"
#include "Opencv_mxyBook.h"

int main(int argc, char *argv[])
{
	Mat src = imread("D:/opencvMdl/opencv_tutorial_data-master/images/wm.jpg");
	VideoCapture vsrc("D:/opencvMdl/3D批量布线.wmv");
	VideoCapture vsrcCama(0);
	//namedWindow("testwindow", WINDOW_AUTOSIZE);
	imshow("input", src);
	//CImgConvert convert;
	//convert.bitwise_demo(src);
	//write your code
	Opencv_mxyBook booktest;
	booktest.test_avi_read(vsrc);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
