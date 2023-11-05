#include "ImgConvert.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
using namespace std;

void CImgConvert::testMain()
{
	document_scanner();
}

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

	Mat m3 = Mat::zeros(Size(8, 8), CV_8UC3);
	m3 = Scalar(0,255,0);
	std::cout << "w:" << m3.cols << "h:" << m3.rows << "channls:" << m3.channels() << std::endl;
	std::cout << m3 << std::endl;

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


void CImgConvert::warp()
{
	Mat img = imread("../models/Resources/cards.jpg");
	float w = 250;
	float h = 350;
	Point2f src[4] = { {529,142},{771,190},{405,395},{674,457} };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };
	Mat matrix = getPerspectiveTransform(src, dst);
	Mat imgWarp;
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	for (int i = 0; i < 4; i++)
	{
		circle(img, src[i], 10, Scalar(0, 0, 255), FILLED);
	}

	imshow("img:", img);
	imshow("imgwarp:", imgWarp);
	waitKey(0);
}

void CImgConvert::colorDectection()
{
	Mat img = imread("../models/Resources/shapes.png");

	Mat imgHsv;
	cvtColor(img, imgHsv, COLOR_BGR2HSV);

	int hmin = 0;
	int smin = 110;
	int vmin = 153;
	int hmax = 19;
	int smax = 240;
	int vmax = 255;

	namedWindow("trackbar", (640, 200));
	createTrackbar("hue min", "trackbar", &hmin, 179);
	createTrackbar("hue max", "trackbar", &hmax, 179);
	createTrackbar("sat min", "trackbar", &smin, 255);
	createTrackbar("sat max", "trackbar", &smax, 255);
	createTrackbar("val min", "trackbar", &vmin, 255);
	createTrackbar("val max", "trackbar", &vmax, 255);
	Mat mask;

	while (1)
	{
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);
		inRange(imgHsv, lower, upper, mask);

		imshow("img", img);
	//	imshow("img hsv", imgHsv);
		imshow("img mask", mask);
		waitKey(1);
	}
	waitKey(0);
}

void CImgConvert::contourDectection()
{
	Mat img = imread("../models/Resources/shapes.png");
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	//GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgGray, imgCanny, 25, 75);
	Mat kernnel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernnel);
	getContours(imgDil, img);
	imshow("img", img);
	//imshow("imgGray", imgGray);
	//imshow("imgBlur", imgBlur);
	//imshow("imgCanny", imgCanny);
	//imshow("imgDil", imgDil);
	waitKey(0);
}

void CImgConvert::getContours(Mat imgDil, Mat img)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	findContours(imgDil, contours,hierancy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(img, contours, -1, Scalar(255,0,255), 2);

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		cout << area << endl;
		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP()
			drawContours(img, contours, i, Scalar(255, 0, 255), 2);
		}
	}
}

void CImgConvert::shapeDectection()
{
	Mat img = imread("../models/Resources/shapes.png");
	Mat imgGray, imgBlur, imgCanny, imgDil;

	//预处理
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgGray, imgCanny, 25, 75);
	Mat kernnel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernnel);

	getContours(imgDil, img);

	imshow("img", img);
// 	imshow("img gary", imgGray);
// 	imshow("img blur", imgBlur);
// 	imshow("img canny", imgCanny);
// 	imshow("img Dil", imgDil);
	waitKey(0);
}



void CImgConvert::faceDectection()
{
	string path = "../models/Resources/pedestrian.png";
	Mat img = imread(path);

	CascadeClassifier faceCascade;
	faceCascade.load("../models/Resources/haarcascade_frontalface_default.xml");
	if (faceCascade.empty())
	{
		cout << "load xml error";
	}

	vector<Rect> faces;
	faceCascade.detectMultiScale(img, faces, 1.1, 10);

	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(img, faces[i], Scalar(255, 0, 255), 3);
	}

	imshow("img: ", img);
	waitKey(0);
}




void CImgConvert::document_scanner()
{
	string path = "../models/Resources/paper.jpg";
	Mat img = imread(path);

	resize(img, img, Size(), 0.5, 0.5);

	//1.预处理
	Mat ret = preProcess(img);
	//2. 获取轮廓



	imshow("img: ", img);
	imshow("ret: ", ret);
	waitKey(0);
}

Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;
cv::Mat CImgConvert::preProcess(Mat img)
{
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernnel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernnel);
	//erode(imgDil, imgErode, kernnel);
	return imgDil;
}

void CImgConvert::add_test()
{
	Mat src1(cv::Size(100, 100), CV_8UC1, cv::Scalar(100));
	Mat src2(cv::Size(100, 100), CV_8UC1, cv::Scalar(200));
	Mat dst = Mat::zeros(Size(100, 100), CV_8UC1);
	add(src1, src2, dst);
	imshow("dst:", dst);
	waitKey(0);
}

void CImgConvert::addWeighted_test()
{
	Mat src1(cv::Size(200, 200), CV_8UC3, Scalar(0, 0, 0));
	Mat src2(cv::Size(200, 200), CV_8UC3, Scalar(0, 0, 0));

	putText(src1, "hello", Point(10, 60), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(0, 0, 255));
	putText(src2, "world", Point(10, 120), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(255, 0, 0));

	Mat dst;
	addWeighted(src1, 0.7, src2, 0.3, 0, dst);

	imshow("src1", src1);
	imshow("src2", src2);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::addLogo()
{
	Mat img = imread("../models/mxy/canny.jpg");
	Mat logo = imread("../models/mxy/a.png");

	int rows = logo.rows;
	int cols = logo.cols;
	int channel = logo.channels();

	//获取ROI
	Mat imgRoi = Mat(img, Rect(0, 0, cols, rows));

	Mat imgGray;
	cvtColor(logo, imgGray, COLOR_BGR2GRAY);
	Mat threod;
	threshold(imgGray, threod, 10, 255, THRESH_BINARY);
	Mat mask_inv; //相反掩码
	bitwise_not(threod, mask_inv);

	Mat imgand;
	bitwise_and(imgRoi, imgRoi, imgand, mask_inv);
	
	Mat logoand;
	bitwise_and(logo, logo, logoand);

	Mat roiand;
	add(imgand, logoand, roiand);

	

	//设置到roi 方式1
	roiand.copyTo(imgRoi);
	//方式2
	//Mat dst;
	//addWeighted(imgRoi, 0, roiand, 1, 0, imgRoi);

	imshow("roiand", img);
	waitKey(0);
}

void CImgConvert::colorspace()
{

	Scalar lower(110, 50, 50);
	Scalar upper(130, 255, 255);

	Mat img = imread("../models/mxy/canny.jpg");

	//转成HSV
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, lower, upper, mask);

	//获取到颜色和原图相加
	Mat dst;
	bitwise_and(img, img, dst, mask);

	imshow("img", img);
	imshow("mask", mask);
	imshow("dst", dst);
	waitKey(0);

}

void CImgConvert::resize_test()
{
	Mat img = imread("../models/mxy/a.png");

	Mat dst;
	resize(img, dst, Size(0, 0), 2, 2, INTER_CUBIC);

	imshow("dst", dst);
	waitKey(0);

}

void CImgConvert::move()
{
	//平移矩阵
	//    1  0  tx
	//	  0  1  ty
	Mat img = imread("../models/mxy/a.png");

	Mat dst;

	//Mat matrix = Mat(Size(3, 2), CV_32FC1);
	//这里int类型会崩溃
 	Mat matrix = (Mat_<float>(2, 3) << 1, 0, 50, 0, 1, 50);
// 	for (int r = 0; r < marix.rows; r++) {
// 		for (int c = 0; c < marix.cols; c++) {
// 			cout << marix.at<int>(r, c) << ",";
// 		}
// 		cout << endl;
// 	}
// 	matrix.at<float>(0, 0) = 1;
// 	matrix.at<float>(0, 2) = 50; //水平平移量
// 	matrix.at<float>(1, 1) = 1;
// 	matrix.at<float>(1, 2) = 0; //竖直平移量

	warpAffine(img, dst, matrix, Size(img.cols, img.rows));

	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::rotate()
{

	#define pi 3.1415926
	// http://www.woshicver.com/FifthSection/4_2_%E5%9B%BE%E5%83%8F%E5%87%A0%E4%BD%95%E5%8F%98%E6%8D%A2/
	Mat img = imread("../models/mxy/a.png");

	Mat dst;
	Mat dst1;
// 	Mat matrix = (Mat_<float>(2, 3) << cos(90 * pi/ 180), -sin(90* pi / 180),
// 									   sin(90* pi / 180), cos(90* pi / 180));

	Point2f pt((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	Mat matrix1 =  getRotationMatrix2D(pt, 90, 1);

	//warpAffine(img, dst, matrix, Size(img.cols, img.rows));
	warpAffine(img, dst1, matrix1, Size(img.cols, img.rows));

	imshow("dst", dst);
	imshow("dst", dst1);
	waitKey(0);
}

void CImgConvert::Affine()
{
	Mat img = imread("../models/mxy/a.png");

	Mat dst1;

	Point2f src[3] = { { 10.0f,10.0f },{ 50.0f,10.0f },{ 10.0f,50.0f } };
	Point2f dst[3] = { { 50.0f,30.0f },{ 90.0f,40.0f },{ 20.0f,60.0f } };

	Mat matrix = getAffineTransform(src, dst);

	warpAffine(img, matrix, dst1, Size(img.cols, img.rows));
	imshow("dst", dst1);
	waitKey(0);
}


void CImgConvert::filter2D_test()
{
	Mat img = imread("../models/mxy/lena.png");

	//Mat kernel = Mat::ones(Size(5, 5), CV_8UC1)/25;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);

	Mat dst;
	filter2D(img, dst, -1, kernel);

	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::blur_test()
{
	Mat img = imread("../models/mxy/lena.png");
	Mat dst;
	blur(img, dst, Size(5, 5));
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::GaussianBlur_test()
{
	Mat img = imread("../models/mxy/lena.png");
	Mat dst;
	GaussianBlur(img, dst, Size(5, 5), 0);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::erode_test()
{
	Mat img = imread("../models/mxy/a.png");
	Mat dst;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	erode(img, dst, kernel, Point(-1, -1), 2);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::dilate_test()
{
	Mat img = imread("../models/mxy/a.png");
	Mat dst;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(img, dst, kernel, Point(-1, -1), 2);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::open()
{
	Mat img = imread("../models/mxy/cells.png");
	Mat dst;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(img, dst, MORPH_OPEN, kernel);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::close()
{
	Mat img = imread("../models/mxy/cells.png");
	Mat dst;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(img, dst, MORPH_CLOSE, kernel);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::graint_test()
{
	Mat img = imread("../models/mxy/lena.png");
	Mat dst;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(img, dst, MORPH_GRADIENT, kernel);
	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::Sobel_laplacian_test()
{
	Mat img = imread("../models/mxy/coins.jpg");
	Mat dst;
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat sobelx,sobely;
	Sobel(img, sobelx, -1, 1, 0, 5);
	Sobel(img, sobely, -1, 0, 1, 5);
	Laplacian(img, dst, -1);
	imshow("img", img);
	imshow("sobelx", sobelx);
	imshow("sobely", sobely);
	imshow("dst", dst);
	waitKey(0);
}

void CImgConvert::canny_test()
{
	Mat img = imread("../models/mxy/coins.jpg");
	Mat dst;
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat edges;
	Canny(img, edges, 100, 200);
	imshow("img", img);
	imshow("dst", edges);
	waitKey(0);
}




class LaplacianBlending {
private:
	Mat leftImg;			//左图
	Mat rightImg;			//右图
	Mat blendMask;			//融合所需要的mask

							//Laplacian Pyramids  左图、右图、结果图拉普拉斯金字塔
	vector<Mat> leftLapPyr, rightLapPyr, resultLapPyr;
	//左图、右图、结果图 最高层图像（也就是最小分辨率图像）
	Mat leftHighestLevel, rightHighestLevel, resultHighestLevel;
	//掩摸mask高斯金字塔 mask为三通道图像，方便矩阵相乘
	vector<Mat> maskGaussianPyramid;

	//层数
	int levels;

	//创建金字塔
	void buildPyramids()
	{
		buildLaplacianPyramid(leftImg, leftLapPyr, leftHighestLevel);
		buildLaplacianPyramid(rightImg, rightLapPyr, rightHighestLevel);
		buildGaussianPyramid();
	}

	//创建高斯金字塔  /金字塔内容为每一层的掩模mask
	void buildGaussianPyramid()
	{
		assert(leftLapPyr.size() > 0);

		maskGaussianPyramid.clear();
		Mat currentImg; 
		cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);
		//保存mask金字塔的每一层图像
		maskGaussianPyramid.push_back(currentImg); //0 - level

		currentImg = blendMask;
		for (int l = 1; l < levels + 1; l++) {
			Mat _down;
			if (leftLapPyr.size() > l)
				pyrDown(currentImg, _down, leftLapPyr[l].size());
			else
				pyrDown(currentImg, _down, leftHighestLevel.size()); //lowest level

			Mat down;
			cvtColor(_down, down, COLOR_GRAY2BGR);
			//add color blend mask into mask Pyramid
			maskGaussianPyramid.push_back(down);
			string winName = to_string((long long)l);
			imshow(winName, down);
			//			waitKey(0);
			currentImg = _down;
		}
	}

	//创建拉普拉斯金字塔
	void buildLaplacianPyramid(const Mat& img, vector<Mat>& lapPyr, Mat& HighestLevel)
	{
		lapPyr.clear();
		Mat currentImg = img;
		for (int l = 0; l < levels; l++) {
			Mat down, up;
			pyrDown(currentImg, down);
			pyrUp(down, up, currentImg.size());
			Mat lap = currentImg - up;
			lapPyr.push_back(lap);
			currentImg = down;
		}
		currentImg.copyTo(HighestLevel);
	}

	//重建图片 从 拉普拉斯金字塔中
	Mat reconstructImgFromLapPyramid()
	{
		//将左右laplacian图像拼成的resultLapPyr金字塔中每一层
		//从上到下插值放大并与残差相加，即得blend图像结果
		Mat currentImg = resultHighestLevel;
		for (int l = levels - 1; l >= 0; l--)
		{
			Mat up;
			pyrUp(currentImg, up, resultLapPyr[l].size());
			currentImg = up + resultLapPyr[l];
		}
		return currentImg;
	}

	//混合拉普拉斯金字塔
	//获得每层金字塔中直接用左右两图Laplacian，变换拼成的图像resultLapPyr（结果拉普拉斯金字塔）
	void blendLapPyrs()
	{
		//结果拉普拉斯金字塔 最高层 混合
		resultHighestLevel = leftHighestLevel.mul(maskGaussianPyramid.back()) +
			rightHighestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());

		//结果拉普拉斯金字塔 除最高层以外的其他层混合
		for (int l = 0; l < levels; l++)
		{
			Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
			Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
			Mat B = rightLapPyr[l].mul(antiMask);
			Mat blendedLevel = A + B;

			resultLapPyr.push_back(blendedLevel);
		}
	}

public:
	LaplacianBlending(const Mat& _left, const Mat& _right, const Mat& _blendMask, int _levels) :  //construct function, used in LaplacianBlending lb(l,r,m,4);
		leftImg(_left), rightImg(_right), blendMask(_blendMask), levels(_levels)
	{
		assert(_left.size() == _right.size());
		assert(_left.size() == _blendMask.size());
		//创建拉普拉斯金字塔和高斯金字塔
		buildPyramids();
		//每层金字塔图像合并为一个
		blendLapPyrs();
	};

	Mat blend()
	{
		//重建拉普拉斯金字塔
		return reconstructImgFromLapPyramid();
	}
};

void CImgConvert::pyramid_test()
{
	Mat apple = imread("../models/mxy/apple.jpg");
	Mat orages = imread("../models/mxy/orages.jpg");
	resize(orages, orages, Size(apple.cols, apple.rows));

// 	Mat applemin;
// 	pyrDown(apple, applemin);
// 
// 	Mat oragesmin;
// 	pyrDown(orages, oragesmin);
// 
// 	Mat apple_up;
// 	pyrUp(applemin, apple_up, Size(apple.cols, apple.rows));
// 
// 	Mat lapla_apple = apple - apple_up;
// 	
// 	Mat orages_up;
// 	pyrUp(oragesmin, orages_up, Size(orages.cols, orages.rows));
// 
// 	Mat lapla_orages = orages - orages_up;
	Mat leftImg32f, rightImg32f;
	apple.convertTo(leftImg32f, CV_32F);
	orages.convertTo(rightImg32f, CV_32F);

	Mat mask = Mat::zeros(leftImg32f.rows, leftImg32f.cols, CV_32FC1);
	//所有行，  列的一半
	mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;

	LaplacianBlending laplaceBlend(leftImg32f, rightImg32f, mask, 10);
	Mat blendImg = laplaceBlend.blend();
	blendImg.convertTo(blendImg, CV_8UC3);
	Mat blendImg_min;
	pyrDown(blendImg, blendImg_min);
	//imshow("apple", apple);
	//imshow("applemin", applemin);
	//imshow("apple_up", apple_up);
	//imshow("lapla_apple", lapla_apple);
	imshow("blendImg_min", blendImg_min);
	waitKey(0);
}

void CImgConvert::contour_test()
{
	Mat img = imread("../models/mxy/a.png");

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat threod;
	threshold(gray, threod, 10, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	//simple 用部分点 不需要太连续 删除冗余节省内存
	findContours(threod, contours, hierancy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i %2 == 0)
		{
			drawContours(img, contours, i, Scalar(0, 255, 0), 3);
		}
		else {
			drawContours(img, contours, i, Scalar(0, 0, 255), 3);
		}
	}
	
	imshow("img", img);
	//imshow("contours", contours);
	waitKey(0);
}

void CImgConvert::contour_feattest()
{
	Mat img = imread("../models/mxy/a.png");

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat threod;
	threshold(gray, threod, 10, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	//simple 用部分点 不需要太连续 删除冗余节省内存
	findContours(threod, contours, hierancy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i % 2 == 0)
		{
			drawContours(img, contours, i, Scalar(0, 255, 0), 3);
		}
		else {
			drawContours(img, contours, i, Scalar(0, 0, 255), 3);
		}
		//轮廓面积
		double area = contourArea(contours[i]);
		//周长
		double len = arcLength(contours[i], true);
		cout << "area: " << area << endl;
		cout << "len: " << len << endl;
	}
	//计算特征矩
	//Moments mom = moments(contours[0]);
	
	
	imshow("img", img);
	waitKey(0);
}

void CImgConvert::convexhull()
{
	Mat img = imread("../models/mxy/a.png");

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat threod;
	threshold(gray, threod, 10, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	//simple 用部分点 不需要太连续 删除冗余节省内存
	findContours(threod, contours, hierancy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i % 2 == 0)
		{
			drawContours(img, contours, i, Scalar(0, 255, 0), 3);
		}
		else {
			drawContours(img, contours, i, Scalar(0, 0, 255), 3);
		}
		//轮廓面积
		double area = contourArea(contours[i]);
		//周长
		double len = arcLength(contours[i], true);
		cout << "area: " << area << endl;
		cout << "len: " << len << endl;
		vector<Point> hull;
		convexHull(contours[i], hull);
		polylines(img, hull, true, Scalar(0, 255, 0), 1, LINE_AA);
	}


	imshow("img", img);
	waitKey(0);
}

void CImgConvert::boundingbox()
{
	Mat img = imread("../models/mxy/a.png");

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat threod;
	threshold(gray, threod, 10, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	//simple 用部分点 不需要太连续 删除冗余节省内存
	findContours(threod, contours, hierancy, RETR_TREE, CHAIN_APPROX_SIMPLE);


	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i % 2 == 0)
		{
			drawContours(img, contours, i, Scalar(0, 255, 0), 3);
		}
		else {
			drawContours(img, contours, i, Scalar(0, 0, 255), 3);
		}
		
		//直角矩形
		Rect boundrect = boundingRect(contours[i]);
		RotatedRect roateRect = minAreaRect(contours[i]);
		
		Point2f vtx[4];
		roateRect.points(vtx);

		// Draw the bounding box
		for (i = 0; i < 4; i++)
			line(img, vtx[i], vtx[(i + 1) % 4], Scalar(255, 0, 0), 1, LINE_AA);

		rectangle(img, boundrect, Scalar(255, 0, 0), 2);
	}

	imshow("img", img);
	waitKey(0);
}

void CImgConvert::hist_test()
{
	Mat img = imread("../models/mxy/coins.jpg");
		
	Mat hist;
	int channels[1];
	channels[0] = 0;
	int histSize[1];
	histSize[0] = 256;
	float hranges[2];         // 值范围
	const float* ranges[1]; // 值范围的指针
	hranges[0] = 0.0;       // 从0开始（含）
	hranges[1] = 256.0;     // 到256（不含）
	ranges[0] = hranges;
	cv::calcHist(&img, 1,    // 仅为一幅图像的直方图
		channels,     // 使用的通道
		cv::Mat(),    // 不使用掩码
		hist,          // 作为结果的直方图
		1,              // 这是一维的直方图
		histSize,     // 箱子数量
		ranges         // 像素值的范围
		);


	// 取得箱子值的最大值和最小值
	double maxVal = 0;
	double minVal = 0;
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

	// 取得直方图的大小
	int zoom = 1;
	int histRow = hist.rows;
	// 用于显示直方图的方形图像
	cv::Mat histImg(histRow*zoom, histRow*zoom,
		CV_8U, cv::Scalar(255));

	// 设置最高点为90%（即图像高度）的箱子个数
	int hpt = static_cast<int>(0.9*histRow);

	// 为每个箱子画垂直线
	for (int h = 0; h < histRow; h++) {
		float binVal = hist.at<float>(h);
		if (binVal > 0) {
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h*zoom, histRow*zoom),
				cv::Point(h*zoom, (histRow - intensity)*zoom),
				cv::Scalar(0), zoom);
		}
	}

	imshow("img", img);
	imshow("hist", histImg);
	waitKey(0);
}

void CImgConvert::discrete_fourier()
{
	Mat I = imread("../models/mxy/coins.jpg", IMREAD_GRAYSCALE);
	if (I.empty()) {
		cout << "Error opening image" << endl;
		return ;
	}
	Mat padded; //放大图片尺寸到合适的大小
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI); // Add to the expanded another plane with zeros
	dft(complexI, complexI); // this way the result may fit in the source matrix
							 // compute the magnitude and switch to logarithmic scale
							 // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	magI += Scalar::all(1); // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy)); // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy)); // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											  // viewable image form (float between values 0 and 1).
	imshow("Input Image", I); // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();
}

void CImgConvert::match_template()
{
	Mat img = imread("../models/mxy/model.jpg");
	Mat head = imread("../models/mxy/model_head.jpg");
	Mat src = img.clone();

	TemplateMatchModes modes[] = { TM_SQDIFF ,
		TM_SQDIFF_NORMED ,
		TM_CCORR ,
		TM_CCORR_NORMED,
		TM_CCOEFF,
		TM_CCOEFF_NORMED };
	double w, h;
	w = head.cols;
	h = head.rows;
	Point top_left, bottom_right;
	for (int i = 0; i < 6; i++)
	{
		Mat temimg = src.clone();
		Mat out;
		matchTemplate(temimg, head, out,  modes[i]);
		
		double min_val, max_val;
		Point min_loc, max_loc;
		minMaxLoc(out, &min_val, &max_val, &min_loc, &max_loc);
		if (modes[i] == TM_SQDIFF || modes[i] == TM_SQDIFF_NORMED)
		{
			top_left = min_loc;
		}else{
			top_left = max_loc;
		}
		bottom_right = Point(top_left.x + w, top_left.y + h);

		//绘制矩形
		rectangle(temimg, top_left, bottom_right, 255, 2);

		std::string title = "out";
		title += to_string(i);
		imshow(title, temimg);
	}
	waitKey();
}

void CImgConvert::watershed_test()
{
	//分割相互分割的对象
	Mat img = imread("../models/mxy/coins.jpg");
	int col = img.cols;
	int row = img.rows;

	Mat grayImage;
	cvtColor(img, grayImage, COLOR_BGR2GRAY);
	
	//Mat threod;
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel, Point(-1, -1), 3);

	//4. 距离变换
	distanceTransform(grayImage, grayImage, DIST_L2, DIST_MASK_3, 5);
	//5. 将图像归一化到[0, 1]范围
	normalize(grayImage, grayImage, 0, 1, NORM_MINMAX);
	//6. 将图像取值范围变为8位(0-255)
	grayImage.convertTo(grayImage, CV_8UC1);
	//7. 再使用大津法转为二值图，并做形态学闭合操作
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
	//8. 使用findContours寻找marks
	vector<vector<Point>> contours;
	findContours(grayImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Mat marks = Mat::zeros(grayImage.size(), CV_32SC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		//static_cast<int>(i+1)是为了分水岭的标记不同，区域1、2、3...这样才能分割
		drawContours(marks, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i + 1)), 2);
	}
	//9. 对原图做形态学的腐蚀操作
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(img, img, MORPH_ERODE, k);
	//10. 调用opencv的分水岭算法
	watershed(img, marks);
	//11. 随机分配颜色
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// 12. 显示
	Mat dst = Mat::zeros(marks.size(), CV_8UC3);
	int index = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			index = marks.at<int>(i, j);
			if (index > 0 && index <= contours.size()) {
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
			else if (index == -1)
			{
				dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else {
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}
	imshow("dst", dst);
	//imwrite("dst.jpg", dst);

	//imshow("threod", threod);
	waitKey();
}
