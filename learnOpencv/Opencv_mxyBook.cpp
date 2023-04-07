#include "Opencv_mxyBook.h"



Opencv_mxyBook::Opencv_mxyBook()
{
}


Opencv_mxyBook::~Opencv_mxyBook()
{
}

void Opencv_mxyBook::DrawEllipse(Mat img, double angle)
{
	int thick = 2;
	int lineType = 8;
	const int window_width = 600;
	ellipse(img, Point(window_width / 2, window_width / 2),
		Size(window_width/4,window_width/16),
		angle,
		0,360,
		Scalar(255,199,0),
		thick,
		lineType);
}

void Opencv_mxyBook::DrawFilledCircle(Mat img, Point center)
{
	int thick = 2;
	int lineType = 8;
	circle(img, center, 600 / 32, Scalar(0,0,255),thick,lineType);
}

void Opencv_mxyBook::DrawLine(Mat img, Point start, Point end)
{
	int thick = 2;
	int lineType = 8;
	line(img,start,end,Scalar(0,0,0),thick,lineType);
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

void Opencv_mxyBook::test_img_add(Mat &image)
{
	Mat sonImg;
	sonImg = imread("D:/opencvMdl/opencv_tutorial_data-master/images/9_99.png");

	//imshow("父图片", image);

	//namedWindow("子图片");
	//imshow("子图片", sonImg);

	//对image中  0 ， 0 区域的引用。
	Mat imageROI;
	imageROI = image(Rect(0, 0, sonImg.cols, sonImg.rows));

	//imshow("结果", imageROI);

	//这块区域 加上sonimg 结果输出到这块区域。 所以会覆盖父图片的0，0区域
	addWeighted(imageROI, 0.5, sonImg, 0.3, 0, imageROI);

	imshow("结果", image);
}

void Opencv_mxyBook::test_img_drwGeom(Mat &src)
{
	Mat mat = Mat::zeros(Size(600, 600), CV_8UC3);
	//绘制椭圆
	DrawEllipse(mat, 90);
	DrawEllipse(mat, 0);
	DrawEllipse(mat, 45);
	DrawEllipse(mat, -45);
	//绘制圆心
	DrawFilledCircle(mat,Point(600/2,600/2));
	//DrawFill
	imshow("绘制几何", mat);
}
