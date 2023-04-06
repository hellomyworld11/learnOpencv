#include "Opencv_mxyBook.h"



Opencv_mxyBook::Opencv_mxyBook()
{
}


Opencv_mxyBook::~Opencv_mxyBook()
{
}

void Opencv_mxyBook::test_img_structur(Mat &image)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat dstimg;
	erode(image,dstimg,element);
	imshow("腐蚀结果:", dstimg);
}

void Opencv_mxyBook::test_img_vague(Mat &src)
{
	Mat dstImg;
	blur(src,dstImg,Size(7,7));
	imshow("均值滤波:", dstImg);
}

void Opencv_mxyBook::test_avi_read(VideoCapture &src)
{
	while (1)
	{
		Mat frame;
		src >> frame;
		Mat dstImg;
		blur(frame, dstImg, Size(7, 7));
		imshow("读取视频", dstImg);
		waitKey(30);
	}
}
