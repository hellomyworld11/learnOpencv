#include "LearnOpencv3.h"



LearnOpencv3::LearnOpencv3()
{
}


LearnOpencv3::~LearnOpencv3()
{
}

void LearnOpencv3::smoothing()
{
	cv::Mat readImg =  cv::imread("../models/stuff.jpg");
	if (readImg.empty())
	{
		std::cout << "read error\n";
		return;
	}
	cv::namedWindow("image-in", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("image-out", cv::WINDOW_AUTOSIZE);
	
	cv::imshow("image-in", readImg);

	cv::Mat out;
	cv::GaussianBlur(readImg, out, cv::Size(5, 5), 3, 3);
	cv::GaussianBlur(out, out, cv::Size(5, 5), 3, 3);

	cv::imshow("image-out", out);

	cv::waitKey(0);
}
