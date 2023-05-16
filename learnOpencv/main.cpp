#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#include "ImgConvert.h"
#include "Opencv_mxyBook.h"
#include "LearnOpencv.h"

int main(int argc, char *argv[])
{
	//Mat src = imread("D:/opencvMdl/opencv_tutorial_data-master/images/wm.jpg");
	//VideoCapture vsrc("D:/opencvMdl/3D批量布线.wmv");
	//VideoCapture vsrcCama(0);
	//namedWindow("testwindow", WINDOW_AUTOSIZE);
	//imshow("input", src);
	//CImgConvert convert;
	//convert.mat_creation_demo(src);
	//write your code
	//Opencv_mxyBook booktest;
	//booktest.test_img_add(src);
	//booktest.test_img_drwGeom(src);
	CLearnOpencv lp;
	lp.mat_load_show_save();

	
	destroyAllWindows();
	return 0;
}
