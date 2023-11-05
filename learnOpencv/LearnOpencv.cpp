#include "LearnOpencv.h"
#include <opencv2/imgproc.hpp>


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


	//水平翻转
	cv::Mat result;
	cv::flip(image, result, 1);
	cv::namedWindow("Output image");
	cv::imshow("Output image", result);
	cv::setMouseCallback("Output image", onMouse, reinterpret_cast<void*>(&result));
	cv::imwrite("output.bmp", result);
	cv::waitKey(0);
}

void CLearnOpencv::video_Capture()
{
	cv::VideoCapture video;
	video.open("D:/opencvMdl/01.mp4");
	if (!video.isOpened())
	{
		std::cout << "没打开" << std::endl;
		return;
	}

	std::cout << video.get(cv::VideoCaptureProperties::CAP_PROP_FPS) << std::endl;   // 23 fps
	std::cout << video.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH) << std::endl;

	while (1)
	{
		cv::Mat frame;
		video >> frame;

		if (frame.empty())
		{
			break;
		}
		imshow("video",frame);

		uchar c = cv::waitKey(1000/ video.get(cv::VideoCaptureProperties::CAP_PROP_FPS));
		if (c == 'q')
		{
			break;
		}
	}
	cv::waitKey(0);
}

void CLearnOpencv::camera_capture()
{
	using namespace std;
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "open cam error" << endl;
		getchar();
		return;
	}
	cout << "success" << endl;
	int fps = capture.get(CAP_PROP_FPS);
	if (fps <= 0)
		fps = 25;
	VideoWriter writer;
	writer.open("xupan.avi",
		VideoWriter::fourcc('x', '2', '6', '4'),
		fps, 
		Size(capture.get(CAP_PROP_FRAME_WIDTH), 
			capture.get(CAP_PROP_FRAME_HEIGHT))
		);

	if (writer.isOpened())
	{
		cout << "create avi error" << endl;
		getchar();
		return;
	}

	namedWindow("cam");
	Mat img;
	for (;;)
	{
		capture.read(img);
		if (img.empty())
			break;

		imshow("cam", img);
		if (waitKey(5) == 'q')
			break;
	
	}

}

void CLearnOpencv::mat_drw_pic()
{
	cv::Mat pic = cv::Mat::zeros(cv::Size(100, 100), CV_8UC1);

	cv::Scalar color = cv::Scalar(255, 255, 255);
	double dr = 5;
	double thick = 3;
	cv::circle(pic, cv::Point(50, 50), dr, color, thick);
	cv::putText(pic, "hello world", cv::Point(45, 60), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);

	imshow("pic", pic);

	cv::waitKey(0);
}

void CLearnOpencv::mat_test()
{
	//CV_8U 1 bytes  CV_8UC3 3 bytes
	//创建一个图像  三通道
	cv::Mat image1(3, 3, CV_8UC3, 100);
	std::cout << "image size = " << image1.elemSize() << std::endl;
// 	cv::imshow("Image", image1);
// 	cv::waitKey(0);
// 
// 	image1.create(200, 200, CV_8U);
// 	image1 = 200;
// 
// 	cv::imshow("Image", image1);
// 	cv::waitKey(0);
	
	cv::Mat image2(cv::Size(240, 320), CV_8UC3, cv::Scalar(0, 0, 255));
// 	cv::imshow("Image", image2);
// 	cv::waitKey(0);
// 
// 	cv::Mat image3 = cv::imread("I:\\learnPro\\vspro\\learnOpencv\\models\\fruits.jpg");
// 	cv::Mat image4(image3);
// 	image1 = image3;
// 
// 	image3.copyTo(image2);
// 	cv::Mat image5 = image3.clone();
// 
// 	cv::imshow("Image", image5);
// 	cv::waitKey(0);
// 
// 	cv::flip(image3, image3, 1);
// 
// 	cv::imshow("Image 3", image3);
// 	cv::imshow("Image 1", image1);
// 	cv::imshow("Image 2", image2);
// 	cv::imshow("Image 4", image4);
// 	cv::imshow("Image 5", image5);
// 	cv::waitKey(0);

	cv::Mat gray = cv::Mat(500, 500, CV_8U, 50);

	cv::imshow("Image", gray);
	cv::waitKey(0);

	image1 = cv::imread("I:\\learnPro\\vspro\\learnOpencv\\models\\fruits.jpg",
		cv::ImreadModes::IMREAD_GRAYSCALE);

	image1.convertTo(image2, CV_32F, 1 / 255.0, 0.0);

	cv::imshow("Image", image2);
	cv::waitKey(0);
}

void CLearnOpencv::mat_ROI()
{
	cv::Mat logo;
	logo = cv::imread("D:/opencvMdl/a.png");
	if (logo.empty())
	{
		std::cout << "read error\n";
		return;
	}
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	

	cv::Mat imageROI(image, cv::Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows));

	cv::Mat mask(logo);
	logo.copyTo(imageROI, mask);

	cv::namedWindow("load image");
	cv::imshow("load image", image);
	cv::waitKey(0);
}

void CLearnOpencv::salt_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	salt(image, 3000);
	cv::namedWindow("load image");
	cv::imshow("load image", image);
	cv::waitKey(0);
}

void CLearnOpencv::reduce_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	reduce(image, 64);
	cv::namedWindow("load image");
	cv::imshow("load image", image);
	cv::waitKey(0);
}

void CLearnOpencv::sharpen_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	cv::Mat result;
	sharpen(image, result);
	
	cv::namedWindow("load image");
	cv::imshow("load image", result);
	cv::waitKey(0);
}

void CLearnOpencv::pic_add()
{
	//注意 两个图片必须大小一样才能用addWeighted
	cv::Mat image1;
	image1 = cv::imread("D:/opencvMdl/pill_001.png");
	cv::Mat image2;
	//image2 = cv::Mat::zeros(image1.size(), image1.type());
	image2 = cv::imread("D:/opencvMdl/pill_002.png");
	cv::resize(image2, image2, image1.size());

	cv::Mat result;
	cv::addWeighted(image1, 0.5, image2, 0.5, 0, result);

//	cv::namedWindow("load image");
	cv::imshow("load image", result);
	cv::waitKey(0);
}

void CLearnOpencv::wave_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	cv::Mat result;
	wave(image, result);

	cv::namedWindow("load image");
	cv::imshow("load image", result);
	cv::waitKey(0);
}

void CLearnOpencv::grabcut_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	cv::Mat result;
	grabcut(image, result);

	cv::namedWindow("load image");
	cv::imshow("load image", result);
	cv::waitKey(0);
}

void CLearnOpencv::hispic_test()
{
	cv::Mat image;
	image = cv::imread("D:/opencvMdl/gray.png");
	cv::Mat result;
	calchist(image, result);

	cv::namedWindow("load image");
	cv::imshow("load image", result);
	cv::waitKey(0);
}

void CLearnOpencv::salt(cv::Mat image, int n)
{
	std::default_random_engine  generator;
	std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
	std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

	int i, j;
	for (int k = 0; k < n; k++)
	{
		i = randomCol(generator);
		j = randomRow(generator);
		if (image.type() == CV_8UC1)
		{
			image.at<uchar>(j, i) = 255;
		}
		else if (image.type() == CV_8UC3)
		{
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

void CLearnOpencv::reduce(cv::Mat image, int div /*= 64*/)
{
	int r = image.rows;
	int c = image.cols * image.channels();

	for (int i = 0; i < r; i++)
	{
		uchar* data = image.ptr<uchar>(i);

		for (int j = 0; j < c; j++)
		{
			data[j] = data[j] / div * div + div / 2;
		}
	}
}

void CLearnOpencv::sharpen(const cv::Mat& image, cv::Mat &result)
{
	result.create(image.size(), image.type());
	int channels = image.channels();

	for (int i = 1; i < image.rows - 1; i++)
	{
		const uchar* pre = image.ptr<const uchar>(i - 1);
		const uchar* cur = image.ptr<const uchar>(i);
		const uchar* next = image.ptr<const uchar>(i + 1);

		uchar* output = result.ptr<uchar>(i);

		for (int j = channels; j < (image.cols - 1) * channels ; j++)
		{
			//锐化算子
			*output++ = cv::saturate_cast<uchar>(5 * cur[j] - cur[j - channels] - cur[i + channels] - pre[i] - next[i]);
		}
	}

	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows - 1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void CLearnOpencv::wave(const cv::Mat& image, cv::Mat& result)
{
	cv::Mat srcX(image.rows, image.cols, CV_32F);
	cv::Mat srcY(image.rows, image.cols, CV_32F);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			srcX.at<float>(i, j) = j;
			srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
		}
	}

	cv::remap(image, result, srcX, srcY, cv::INTER_LINEAR);
}

void CLearnOpencv::grabcut(const cv::Mat& image, cv::Mat& result)
{
	cv::Rect rectangle(5, 70, 260, 120);
	cv::Scalar color = cv::Scalar(255, 255, 255);
	double dr = 5;
	double thick = 3;
	cv::rectangle(image, rectangle,  color, thick);
	//cv::Mat result;
	cv::Mat bgModel, fgModel;
	
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	image.copyTo(foreground, result);
	foreground.copyTo(result, result);
}

void CLearnOpencv::calchist(const cv::Mat& image, cv::Mat &result)
{
	// 创建灰度图像的直方图
	class Histogram1D {
	private:
		int histSize[1];          // 直方图中箱子的数量
		float hranges[2];         // 值范围
		const float* ranges[1]; // 值范围的指针
		int channels[1];          // 要检查的通道数量
	public:
		Histogram1D() {

			// 准备一维直方图的默认参数
			histSize[0] = 256;      // 256个箱子
			hranges[0] = 0.0;       // 从0开始（含）
			hranges[1] = 256.0;     // 到256（不含）
			ranges[0] = hranges;
			channels[0] = 0;        // 先关注通道0
		}
		// 计算一维直方图
		cv::Mat getHistogram(const cv::Mat &image) {


			cv::Mat hist;
			// 用calcHist函数计算一维直方图
			cv::calcHist(&image, 1,    // 仅为一幅图像的直方图
				channels,     // 使用的通道
				cv::Mat(),    // 不使用掩码
				hist,          // 作为结果的直方图
				1,              // 这是一维的直方图
				histSize,     // 箱子数量
				ranges         // 像素值的范围
				);
			return hist;
		}

		// 计算一维直方图，并返回它的图像
		cv::Mat getHistogramImage(const cv::Mat &image, int zoom = 1) {


			// 先计算直方图
			cv::Mat hist = getHistogram(image);
			// 循环遍历每个箱子
			for (int i = 0; i < 256; i++)
				std::cout << "Value " << i << " = "
				<< hist.at<float>(i) << std::endl;
			// 创建图像
			return getImageOfHistogram(hist, zoom);
		}

		// 创建一个表示直方图的图像（静态方法）
		cv::Mat getImageOfHistogram(const cv::Mat &hist, int zoom) {
			// 取得箱子值的最大值和最小值
			double maxVal = 0;
			double minVal = 0;
			cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

			// 取得直方图的大小
			int histSize = hist.rows;

			// 用于显示直方图的方形图像
			cv::Mat histImg(histSize*zoom, histSize*zoom,
				CV_8U, cv::Scalar(255));

			// 设置最高点为90%（即图像高度）的箱子个数
			int hpt = static_cast<int>(0.9*histSize);

			// 为每个箱子画垂直线
			for (int h = 0; h < histSize; h++) {
				float binVal = hist.at<float>(h);
				if (binVal > 0) {
					int intensity = static_cast<int>(binVal*hpt / maxVal);
					cv::line(histImg, cv::Point(h*zoom, histSize*zoom),
						cv::Point(h*zoom, (histSize - intensity)*zoom),
						cv::Scalar(0), zoom);
				}
			}

			return histImg;
		}
	};

	Histogram1D h;
	result = h.getHistogramImage(image);



}
