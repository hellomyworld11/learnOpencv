#include "LearnOpencv.h"



CLearnOpencv::CLearnOpencv()
{
}


CLearnOpencv::~CLearnOpencv()
{
}

void onMouse(int event, int x, int y, int flags, void *userdata)
{
	cv::Mat *im = reinterpret_cast<cv::Mat*>(userdata);
//	int ivalue = im->at<uchar>(cv::Point(x, y));
//	std::cout << "x:" << x << "y:" << y << "pix" << ivalue << std::endl;
	
	if (im == nullptr)
	{
		return;
	}
	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		std::cout << "at(" << x << "," << y << ") value is: " << static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
		break;
	default:
		break;
	}
}


void CLearnOpencv::mat_load_show_save()
{
	cv::Mat image;
	std::cout << "This image is " << image.rows << "X" << image.cols << std::endl;
//	cv::imshow("create mat", image);
	image = cv::imread("D:/opencvMdl/test1.png");
	if (image.empty())
	{
		std::cout << "read error\n";
		return;
	}
	cv::namedWindow("load image");
	cv::imshow("load image", image);

	cv::setMouseCallback("load image", onMouse, reinterpret_cast<void*>(&image));


	//Ë®Æ½·­×ª
	cv::Mat result;
	cv::flip(image, result, 1);
	cv::namedWindow("Output image");
	cv::imshow("Output image", result);

	cv::imwrite("output.bmp", result);
	cv::waitKey(0);
}
