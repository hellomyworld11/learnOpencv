#include "Opencv_mxyBook.h"

using namespace std;


Opencv_mxyBook::Opencv_mxyBook()
{
}


Opencv_mxyBook::~Opencv_mxyBook()
{
}

void Opencv_mxyBook::testMain()
{
	harris_dec();
}

void Opencv_mxyBook::camera_video()
{
	VideoCapture capture(0);
	Mat edges;
	while (1)
	{
		Mat frame;
		capture >> frame;

		//
		cvtColor(frame, edges, COLOR_BGR2GRAY);

		blur(edges, edges, Size(7, 7));

		Canny(edges, edges, 0, 30, 3);
		imshow("read", edges);
		if (waitKey(30) >= 0)break;
	}
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

void Opencv_mxyBook::Roi_Addimage()
{
	Mat img = imread("../models/mxy/lena.png");
	Mat logo_img = imread("../models/mxy/a.png");

	if (!img.data || !logo_img.data)
	{
		std::cout << "read  error" << std::endl;
		return;
	}
	
	Mat roiImg = img(Rect(200, 250, logo_img.cols, logo_img.rows));
	Mat mask = imread("../models/mxy/a.png", IMREAD_GRAYSCALE);

	logo_img.copyTo(roiImg, mask);

	imshow("roi add image", img);

	waitKey(0);
}

void Opencv_mxyBook::Linear_Blending()
{
	double aplhaValue = 0.5;
	double betaValue;

	Mat rainImg, moguImg, dstImg;

	moguImg = imread("../models/mxy/mogu.jpg");
	rainImg = imread("../models/mxy/rain.jpg");

	if (!moguImg.data || !rainImg.data)
	{
		std::cout << "read  error" << std::endl;
		return;
	}

	betaValue = (1.0  - aplhaValue);
	addWeighted(moguImg, aplhaValue, rainImg, betaValue, 0.0, dstImg);

	

	imshow("src img", moguImg);
	imshow("rain img", rainImg);
	imshow("dst img", dstImg);

	waitKey(0);
}

typedef struct UserData {
	int cv;
	int lv;
	Mat img;
	Mat dst;
};

void ContrastTrackbarCallback(int pos, void* userdata)
{
	namedWindow("src", WINDOW_AUTOSIZE);
	UserData userd = *(UserData*)userdata;
	for (int y = 0; y < userd.img.rows; y++)
	{
		for (int x = 0; x < userd.img.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				userd.dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>( (userd.cv * 0.01) *(userd.img.at<Vec3b>(y, x)[c]) + userd.lv  );
			}
		}
	}
	imshow("src", userd.img);
	imshow("result", userd.dst);
}

void LightTrackbarCallback(int pos, void* userdata)
{

}

void Opencv_mxyBook::pic_ligth_contrast_control()
{
	Mat srcImg = imread("../models/mxy/1.jpg");

	if (!srcImg.data)
	{
		std::cout << "imread error" << std::endl;
		return;
	}

	Mat dstImg = Mat::zeros(srcImg.size(), srcImg.type());

	//light value setting
	int contrastv = 80;
	int lightv = 80;

	namedWindow("result", WINDOW_AUTOSIZE);

	UserData *userdata = new UserData;
	userdata->cv = contrastv;
	userdata->lv = lightv;
	userdata->img = srcImg;
	userdata->dst = dstImg;

	createTrackbar("对比度", "result", &userdata->cv, 300, ContrastTrackbarCallback, userdata);
	createTrackbar("亮度", "result", &userdata->lv, 200, ContrastTrackbarCallback, userdata);

	ContrastTrackbarCallback(userdata->cv, userdata);
	ContrastTrackbarCallback(userdata->lv, userdata);

	while (char(waitKey(1)) != 'q') {}
	delete userdata;
}

void Opencv_mxyBook::test_morphology()
{
	Mat srcImg = imread("../models/mxy/1.jpg");
	if (srcImg.empty())
	{
		return;
	}

	namedWindow("src", WINDOW_AUTOSIZE);
	namedWindow("result", WINDOW_AUTOSIZE);

	int size = 30;
	int x = -1;
	int y = -1;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size), Point(x, y));

	imshow("src", srcImg);
	morphologyEx(srcImg, srcImg, MORPH_ERODE, element);

	imshow("result", srcImg);
	waitKey(0);
}

void Opencv_mxyBook::test_canny()
{
	Mat src = imread("../models/mxy/canny.jpg");
	Mat src1 = src.clone();
	imshow("src:", src);

	Mat dst, edge, gray;
	dst.create(src1.size(), src1.type());

	//转灰度图
	cvtColor(src1, gray, COLOR_BGR2GRAY);
	imshow("gray:", gray);
	//滤波
	blur(gray, edge, Size(3, 3));
	imshow("blur:", edge);
	Canny(edge, edge, 3, 9, 3);
	imshow("edge:", edge);
	
	dst = Scalar::all(0);

	src1.copyTo(dst, edge);

	imshow("dst:", dst);
	waitKey(0);
}

void Opencv_mxyBook::test_equalizehist()
{
	Mat src, dst;
	src = imread("../models/mxy/equalhist.jpg");

	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("src: ", src);

	equalizeHist(src, dst);

	imshow("dst: ", dst);
	
	waitKey(0);
}

void Opencv_mxyBook::contour_find()
{
	Mat img = imread("../models/mxy/findcontoure.jpg");
	cvtColor(img, img, COLOR_BGR2GRAY);
	Mat dst = Mat::zeros(img.size(), CV_8UC3 );

	imshow("src: ", img);
	//img 取大于119的部分
	img = img > 119;

	imshow("src>119: ", img);
	vector<vector<Point>> contours;
	vector<Vec4i> hierancy;
	findContours(img, contours, hierancy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


//	findContours(img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	int index = 0;
	for (; index >= 0; index = hierancy[index][0])
	{
		// 0 - 255 随机值
		Scalar color(rand()&255, rand()&255, rand()&255);
		drawContours(dst, contours, index, color, FILLED, 8, hierancy);
	}

	imshow("dst: ", dst);
	waitKey(0);
}

int g_thresh = 80;
int g_thresh_max = 255;

void on_thresh_change(int pos, void *pdata)
{
	Mat img = *(Mat*)pdata;
	Mat outImg;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Canny(img, outImg, g_thresh, g_thresh * 2, 3);

	findContours(outImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	RNG g_rng(12345);
	Mat drawing = Mat::zeros(outImg.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255),
			g_rng.uniform(0, 255), g_rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	imshow("dst: ", drawing);
}

void Opencv_mxyBook::contour_find_dynamic()
{
	string srcwin = "src: ";
	string dstwin = "dst: ";
	Mat img = imread("../models/mxy/Coke.jpg");

	//转灰度
	cvtColor(img, img, COLOR_BGR2GRAY);
	//模糊降噪
	blur(img, img, Size(3, 3));

	namedWindow(srcwin, WINDOW_AUTOSIZE);
	imshow(srcwin, img);
	

	

	createTrackbar("canny 阈值", srcwin, &g_thresh, g_thresh_max, on_thresh_change,
		(void*)&img);

	waitKey(0);
}

void Opencv_mxyBook::converxhull()
{
	Mat img(600, 600, CV_8UC3);
	RNG &rng = theRNG();

	while (1)
	{
		char key;
		int count = (unsigned)rng % 100 + 3;
		vector<Point> points;

		for (int i = 0; i < count; i++)
		{
			Point point;
			point.x = rng.uniform(img.cols / 4, img.cols * 3 / 4);
			point.y = rng.uniform(img.rows / 4, img.rows * 3 / 4);
			points.push_back(point);
		}

		vector<int> hull;

		convexHull(Mat(points), hull, true);
		
		img = Scalar::all(0);
		for (int i = 0; i < count; i++)
		{
			circle(img, points[i], 3, Scalar(rng.uniform(40, 255), rng.uniform(40, 255),
				rng.uniform(40, 255)), FILLED, LINE_AA);
		}

		int hullcount = hull.size();
		Point point0 = points[hull[hullcount-1]];

		for (int i = 0; i < hullcount; i++)
		{
			Point point = points[hull[i]];
			line(img, point0, point, Scalar(255, 255, 255), 2, LINE_AA);
			point0 = point;
		}

		imshow("dst: ", img);

		 key = (char)waitKey();
		if (key == 'q')
		{
			break;
		}
	}

}

void Opencv_mxyBook::picinpaint()
{
	Mat img = imread("../models/mxy/inpant.jpg");
	Mat inpaintMask;
	inpaintMask = Scalar::all(0);
	imshow("src: ", img);

	Mat inpantImg;
	inpaint(img, inpaintMask, inpantImg, 3, INPAINT_TELEA);
	imshow("dst: ", inpantImg);
}

void Opencv_mxyBook::harris_dec()
{
	Mat img = imread("../models/mxy/harris1.jpg");

	imshow("src: ", img);

	Mat dst;
	cornerHarris(img, dst, 2, 3, 0.01);

	Mat harrisCorner;
	threshold(dst, harrisCorner, 0.00001, 255, THRESH_BINARY);
		
	imshow("dst: ", harrisCorner);
	waitKey(0);
}
