#include "ImgConvert.h"




void CImgConvert::colorSpace_demo(Mat &image)
{
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_RGB2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("HSV", hsv);
	imshow("灰度", gray);
	imwrite("D:/opencvMdl/hsv.png",hsv);
	imwrite("D:/opencvMdl/gray.png", gray);
}

void CImgConvert::mat_creation_demo(Mat &image)
{
	Mat dstClone, dstCopy;
	dstClone = image.clone();
	image.copyTo(dstCopy);

	Mat m3 = Mat::zeros(Size(400, 400), CV_8UC3);
	m3 = Scalar(0,255,0);
	std::cout << "w:" << m3.cols << "h:" << m3.rows << "channls:" << m3.channels() << std::endl;
	//std::cout << m3 << std::endl;

	Mat m4 = m3;
	m4 = Scalar(255, 0, 0);

	imshow("create mat", m3);
}

void CImgConvert::pixel_visit_demo(Mat &image)
{
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
// 	for (int i = 0; i < h; i++)
// 	{
// 		for (int j = 0; j < w; j++)
// 		{
// 			if (dims == 1)
// 			{
// 				int pv = image.at<uchar>(i,j);
// 				image.at<uchar>(i, j) = 255 - pv;
// 			}
// 			if (dims == 3)
// 			{
// 				Vec3b bgr = image.at<Vec3b>(i, j);
// 				image.at<Vec3b>(i, j)[0] = 255 - bgr[0];
// 				image.at<Vec3b>(i, j)[1] = 255 - bgr[1];
// 				image.at<Vec3b>(i, j)[2] = 255 - bgr[2];
// 			}
// 		}
// 	}

	for (int i = 0; i < h; i++)
	{
		uchar* current_row = image.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			if (dims == 1)
			{
				int pv = *current_row;
				*current_row++ = 255 - pv;
			}
			if (dims == 3)
			{
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}

	imshow("result", image);
}

void CImgConvert::operators_demo(Mat &image)
{
	Mat dst;
	dst = image + Scalar(50,50,50);
	imshow("add", dst);
}

static void on_track(int initVal, void* userdata) {
	Mat image = *((Mat *)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(initVal, initVal, initVal);
	addWeighted(image,1.0,m,0, initVal,dst);
	imshow("亮度调整", dst);
}

void CImgConvert::tracking_bar_demo(Mat &image)
{
	namedWindow("亮度对比度调整", WINDOW_AUTOSIZE);
	int max_val = 100;
	int lightness = 50;
	int contrast_val = 100;
	createTrackbar("value bar:","亮度对比度调整",&lightness,
		max_val, 
		on_track,
		(void*)&image);
	
	on_track(50, &image);
}

void CImgConvert::color_style_demo(Mat &image)
{
	int colormap[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_JET,
		COLORMAP_WINTER,
		COLORMAP_RAINBOW,
		COLORMAP_OCEAN,
		COLORMAP_SUMMER,
		COLORMAP_SPRING,
		COLORMAP_COOL,
		COLORMAP_PINK,
		COLORMAP_HOT,
		COLORMAP_PARULA,
		COLORMAP_MAGMA,
		COLORMAP_INFERNO,
		COLORMAP_PLASMA,
		COLORMAP_VIRIDIS,
		COLORMAP_CIVIDIS,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED,
	};
	Mat dst;
	int index = 0;
	while (true)
	{
		int c = waitKey(2000);
		if (c == 27)
		{
			break;
		}
		applyColorMap(image, dst, colormap[index % 19]);
		index++;
		imshow("变化", dst);
	}
}

void CImgConvert::bitwise_demo(Mat &image)
{
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1,Rect(100,100,80,80),Scalar(255,255,0),-1,LINE_8,0);
	circle(m2, Point(150, 150),80, Scalar(0, 255, 255),-1,LINE_8,0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	bitwise_or(m1, m2, dst);
	imshow("像素操作", dst);
}
