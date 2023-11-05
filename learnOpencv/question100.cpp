#include "question100.h"
#define _USE_MATH_DEFINES
#include <math.h>


question100::question100()
{
}


question100::~question100()
{
}
cv::Mat gamma_correction1(cv::Mat img, double gamma_c, double gamma_g) {
	// get height and width
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();

	// output image
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);

	double val;

	// gamma correction
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channel; c++) {
				val = (double)img.at<cv::Vec3b>(y, x)[c] / 255;

				out.at<cv::Vec3b>(y, x)[c] = (uchar)(pow(val / gamma_c, 1 / gamma_g) * 255);
			}
		}
	}

	return out;
}
void question100::testMain()
{
	cv::Mat img = cv::imread("../models/que100/imori.jpg", IMREAD_COLOR);

	int width = img.cols;
	int height = img.rows;

	Mat out = afine_transformations_scale(img);
	
	cv::imshow("imori-in", img);
	cv::imshow("imori-out", out);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void question100::tutorial_que()
{
	//交换左半红蓝通道
	cv::Mat img = cv::imread("../models/que100/imori.jpg");

	cv::imshow("imori-in", img);

	int whalf = img.cols / 2;
	int rows = img.rows;

	for (int i = 0; i < whalf; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			uchar temp = img.at<Vec3b>(j, i)[0];
			img.at<Vec3b>(j, i)[0] = img.at<Vec3b>(j, i)[2];
			img.at<Vec3b>(j, i)[2] = temp;		
		}
		
	}

	cv::imshow("imori-out", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

Mat question100::change_channel(Mat img)
{
	int width = img.cols;
	int height = img.rows;

	cv::Mat out = Mat::zeros(Size(width, height), CV_8UC3);

	for (int i = 0; i < img.cols; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			out.at<Vec3b>(j, i)[0] = img.at<Vec3b>(j, i)[2];
			out.at<Vec3b>(j, i)[1] = img.at<Vec3b>(j, i)[1];
			out.at<Vec3b>(j, i)[2] = img.at<Vec3b>(j, i)[0];
		}
	}

	return out;
}

cv::Mat question100::gray_scale(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	double y;
	double rk = 0.2126;
	double gk = 0.7152;
	double bk = 0.0722;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			y = img.at<Vec3b>(j, i)[0] * bk + img.at<Vec3b>(j, i)[1] * gk + img.at<Vec3b>(j, i)[2] * rk;

			img.at<Vec3b>(j, i)[0] = y;
			img.at<Vec3b>(j, i)[1] = y;
			img.at<Vec3b>(j, i)[2] = y;
		}
	}

	return img;
}

cv::Mat question100::thresholding(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	double y;
	double rk = 0.2126;
	double gk = 0.7152;
	double bk = 0.0722;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			y = img.at<Vec3b>(j, i)[0] * bk + img.at<Vec3b>(j, i)[1] * gk + img.at<Vec3b>(j, i)[2] * rk;

			if (y < 128)
			{
				img.at<Vec3b>(j, i)[0] = 0;
				img.at<Vec3b>(j, i)[1] = 0;
				img.at<Vec3b>(j, i)[2] = 0;
			}
			else {
				img.at<Vec3b>(j, i)[0] = 255;
				img.at<Vec3b>(j, i)[1] = 255;
				img.at<Vec3b>(j, i)[2] = 255;
			}
			
		}
	}

	return img;
}

cv::Mat question100::otus_ts(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	double rk = 0.2126;
	double gk = 0.7152;
	double bk = 0.0722;
	double y;
	//Get t
	int t = 0;
	int maxSb = 0;
	double pixs = width*height;
	for (int ti = 0; ti <= 255; ti++)
	{
		double pixnum0 = 0;
		double pixnum1 = 0;
		double w0 = 0, w1 = 0;
		double m0 = 0, m1 = 0;
		int pixv0 = 0, pixv1 = 0;
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				y = img.at<Vec3b>(j, i)[0] * bk + img.at<Vec3b>(j, i)[1] * gk + img.at<Vec3b>(j, i)[2] * rk;

				if (y < ti)
				{
					pixnum0++;
					pixv0 += y;
				}
				else {
					pixnum1++;
					pixv1 += y;
				}

			}
		}
		w0 = pixnum0 / pixs;
		w1 = pixnum1 / pixs;
		if (pixnum0 != 0)
		{
			m0 = pixv0 / pixnum0;
		}
		if (pixnum1 != 0)
		{
			m1 = pixv1 / pixnum1;
		}
			
		if (w0 + w1 != 1)
		{
			//ERROR;
		}
		int Sb = w0 * w1 * pow(m0 - m1, 2);
		if (Sb > maxSb)
		{
			maxSb = Sb;
			t = ti;
		}
	}

	std::cout << "t: " << t << std::endl;

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			y = img.at<Vec3b>(j, i)[0] * bk + img.at<Vec3b>(j, i)[1] * gk + img.at<Vec3b>(j, i)[2] * rk;

			if (y < t)
			{
				img.at<Vec3b>(j, i)[0] = 0;
				img.at<Vec3b>(j, i)[1] = 0;
				img.at<Vec3b>(j, i)[2] = 0;
			}
			else {
				img.at<Vec3b>(j, i)[0] = 255;
				img.at<Vec3b>(j, i)[1] = 255;
				img.at<Vec3b>(j, i)[2] = 255;
			}

		}
	}
	return img;
}

cv::Mat question100::hdv_trans(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	cv::Mat out = Mat::zeros(Size(width, height), CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double r = img.at<Vec3b>(i, j)[2] / 255.0;
			double g = img.at<Vec3b>(i, j)[1] / 255.0;
			double b = img.at<Vec3b>(i, j)[0] / 255.0;
			double rgb[3], hsv[3];
			rgb[0] = r;
			rgb[1] = g;
			rgb[2] = b;
			rgb_to_hsv(rgb, hsv);
			hsv[0] += 180;
			double hnew = fmod(hsv[0], 360);  // H no bigger than 360
			hsv[0] = hnew;   
			hsv_to_rgb(hsv, rgb);
			double r1 = rgb[0] * 255.0;
			double g1 = rgb[1] * 255.0;
			double b1 = rgb[2] * 255.0;
			out.at<Vec3b>(i, j)[2] = r1;
			out.at<Vec3b>(i, j)[1] = g1;
			out.at<Vec3b>(i, j)[0] = b1;
		}
	}
	return out;
}

cv::Mat question100::reduce(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double r = img.at<Vec3b>(i, j)[2];
			double g = img.at<Vec3b>(i, j)[1];
			double b = img.at<Vec3b>(i, j)[0];

			img.at<Vec3b>(i, j)[2] = quantify(r);
			img.at<Vec3b>(i, j)[1] = quantify(g);
			img.at<Vec3b>(i, j)[0] = quantify(b);
		}
	}
	return img;
}

cv::Mat question100::average_pooling(Mat img)
{
	int width = img.cols;
	int height = img.rows;

	double nums = width / 8.0;

	for (int i = 0; i < nums; i++)
	{
		for (int j = 0; j < nums; j++)
		{
			averagev_net(img, i, j, 8);
		}
	}
	return img;
}

cv::Mat question100::max_pooling(Mat img)
{
	int width = img.cols;
	int height = img.rows;

	double nums = width / 8.0;

	for (int i = 0; i < nums; i++)
	{
		for (int j = 0; j < nums; j++)
		{
			max_net(img, i, j, 8);
		}
	}
	return img;
}

cv::Mat question100::gaussian_filter(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	const int kernel_size = 3;
	int sigma = 1.3;

	Mat out = Mat::zeros(height, width, CV_8UC3);

	int pad = floor(kernel_size / 2);
	int _x = 0, _y = 0;
	double kernel_sum = 0;

	float kernel[kernel_size][kernel_size];

	// make kernel matrix
	for (int y = 0; y < kernel_size; y++)
	{
		for (int x = 0; x < kernel_size; x++)
		{
			_x = x - pad;
			_y = y - pad;
			kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp( - (_x * _x + _y * _y) / (2 * sigma * sigma));
			kernel_sum += kernel[y][x];
		}
	}
	
	for (int y = 0; y < kernel_size; y++)
	{
		for (int x = 0; x < kernel_size; x++)
		{
			kernel[y][x] /= kernel_sum;
		}
	}

	//filtering
	double v = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int c = 0; c < channel; c++)
			{
				v = 0;
				for (int dy = -pad; dy < pad + 1; dy++)
				{
					for (int dx = -pad; dx < pad + 1; dx++)
					{
						if ((x + dx) >= 0 && (y + dy) >= 0)
						{
							v += (double)img.at<Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
						}
					}
				}
				out.at<cv::Vec3b>(y, x)[c] = v;
			}
		}

	}

	return out;
}

cv::Mat question100::median_filter(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	int kernel_size = 3;
	int pad = floor(kernel_size / 2);
	double v = 0;
	Mat out = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			
			for (int c = 0; c < channel; c++)
			{
				v = 0;
				vector<double> arr;
				for (int dy = -pad; dy < pad + 1; dy++)
				{
					for (int dx = -pad; dx < pad + 1; dx++)
					{
						if ((j + dx) >= 0 && (i + dy) >= 0)
						{
							arr.push_back((double)img.at<Vec3b>(i + dy, j + dx)[c]);
						}
					}
				}
				v = get_median(arr);
				out.at<cv::Vec3b>(i, j)[c] = v;
			}
		}
	}
	return out;
}

cv::Mat question100::average_filter(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	int kernel_size = 3;
	int pad = floor(kernel_size / 2);
	double v = 0;
	Mat out = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{

			for (int c = 0; c < channel; c++)
			{
				v = 0;
				double all = 0;
				for (int dy = -pad; dy < pad + 1; dy++)
				{
					for (int dx = -pad; dx < pad + 1; dx++)
					{
						if ((j + dx) >= 0 && (i + dy) >= 0)
						{
							all += img.at<Vec3b>(i + dy, j + dx)[c];
						}
					}
				}
				v = all/ (kernel_size * kernel_size);
				out.at<cv::Vec3b>(i, j)[c] = v;
			}
		}
	}
	return out;
}

cv::Mat question100::motion_filter(Mat img)
{

}

cv::Mat question100::histogram(Mat img)
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
	Mat result = h.getHistogramImage(img);
	return result;
}



cv::Mat question100::gamma_correction(Mat img, double gamma_c, double gamma_g)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	
	// output image
	double val;
	cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int c = 0; c < channel; c++) {
				val = (double)img.at<cv::Vec3b>(i, j)[c] / 255;

				out.at<cv::Vec3b>(i, j)[c] = (uchar)(pow(val / gamma_c, 1 / gamma_g) * 255);
			}
		}
	}
	return out;
}

cv::Mat question100::nearest_neiInterpolation(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();

	double scale = 1.5;

	Mat out = Mat::zeros(Size((int)(width * 1.5), (int)(height * 1.5)), CV_8UC3);
	int outwidth = out.cols;
	int outheight = out.rows;

	for (int i = 0; i < outheight; i++)
	{
		int x = (int)round((i / scale));
		x = fmin(x, height - 1);
		for (int j = 0; j < outwidth; j++)
		{		
			int y = (int)round((j / scale));
			y = fmin(y, width - 1);
			for (int c = 0; c < channel; c++)
			{
				out.at<Vec3b>(i, j)[c] = img.at<Vec3b>(x, y)[c];
			}
		}
	}
	return out;
}

cv::Mat question100::afine_transformations_move(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();

	int tx = 30;
	int ty = -30;

	double mat[3][3] = {
		1, 0, tx,
		0, 1, ty,
		0, 0, 1
	};



	Mat out = Mat::zeros(height, width, CV_8UC3);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			vector<int> originpos;
			originpos.push_back(j);
			originpos.push_back(i);
			originpos.push_back(1);
			vector<int> newpos = vecmattrans(mat, originpos);
			if (newpos.size() == 3)
			{
				if (0 <= newpos[0] &&  newpos[0] < width)
				{
					if (0 <= newpos[1] && newpos[1] < height)
					{
						for (int c = 0; c < channel; c++)
						{
							out.at<Vec3b>(newpos[1], newpos[0])[c] = img.at<Vec3b>(i, j)[c];
						}			
					}		
				}		
			}	
		}
	}
	return out;
}

cv::Mat question100::afine_transformations_scale(Mat img)
{
	double xs = 1.3;
	double ys = 4.0 / 5.0;
	double tx = 30;
	double ty = -30;
	double mat[3][3] = {
		xs, 0, tx,
		0,  ys, ty,
		0,  0,   1
	};

	double matinvert[3][3];
	MatrixInvert(mat, matinvert);

	int oldH = img.rows;
	int oldW = img.cols;
	int chnnel = img.channels();
	int newH = (int)oldH * ys;
	int newW = (int)oldW * 1.3;
	Mat out = Mat::zeros(newH,	newW, CV_8UC3);
	
	for (int i = 0; i < newH; i++)
	{
		for (int j = 0; j < newW; j++)
		{
			vector<int> originpos = {j, i, 1};
			vector<int> newpos = vecmattrans(matinvert, originpos);
			if (newpos.size() == 3)
			{
				if (0 <= newpos[0] && newpos[0] < oldW)
				{
					if (0 <= newpos[1] && newpos[1] < oldH)
					{
						for (int c = 0; c < chnnel; c++)
						{
							out.at<Vec3b>(i, j)[c] = img.at<Vec3b>(newpos[1], newpos[0])[c];
						}
					}
				}
			}
		}
	}
	return out;
}

double question100::maxdouble(double a, double b, double c)
{
	return (a > b) ? ( a > c ? a : c ):( b > c ? b : c);
}

double question100::mindouble(double a, double b, double c)
{
	return (a < b) ? (a < c ? a : c) : (b < c ? b : c);
}

double question100::get_hue(double dmin, double dmax, double r, double g, double b)
{
	if (dmin == dmax)
	{
		return 0;
	}
	double d = dmax - dmin;
	if (dmin == b)
	{
		return 60 * ((g - r) / d) + 60;
	}
	if (dmin == r)
	{
		return 60 * ((b - g) / d) + 180;
	}
	if (dmin == g)
	{
		return 60 * ((r - b) / d) + 300;
	}
	return -1;
}

void question100::rgb_to_hsv(double rgb[3], double hsv[3])
{
	double r = rgb[0];
	double g = rgb[1];
	double b = rgb[2];
	double Max = maxdouble(r, g, b);
	double Min = mindouble(r, g, b);
	double hue = get_hue(Min, Max, r, g, b);
	double S = Max - Min;
	double V = Max;
	hsv[0] = hue;
	hsv[1] = S;
	hsv[2] = V;
}

void question100::hsv_to_rgb(double hsv[3], double rgb[3])
{
	double h = hsv[0];
	double s = hsv[1];
	double v = hsv[2];

	double C = s;
	double H1 = h / 60.0;
	double X = C * (1 - abs(fmod(H1, 2) - 1));
	double d = v - C;
	double r, g, b;
	if (0 <= H1 && H1 <= 1)
	{
		r = d + C;
		g = d + X;
		b = d;
	}else if (1 <= H1 && H1 <= 2)
	{
		r = d + X;
		g = d + C;
		b = d;
	}
	else if (2 <= H1 && H1 <= 3)
	{
		r = d ;
		g = d + C;
		b = d + X;
	}
	else if (3 <= H1 && H1 <= 4)
	{
		r = d ;
		g = d + X;
		b = d + C;
	}
	else if (4 <= H1 && H1 <= 5)
	{
		r = d + X;
		g = d ;
		b = d + C;
	}
	else if (5 <= H1 && H1 <= 6)
	{
		r = d + C;
		g = d ;
		b = d + X;
	}
	else {
		r = d;
		g = d;
		b = d;
	}
	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}

double question100::quantify(double value)
{
	if (0 <= value && value < 64)
	{
		return 32;
	}else if (64 <= value && value < 128)
	{
		return 96;
	}
	else if (128 <= value && value < 192)
	{
		return 160;
	}
	else if (192 <= value && value < 256)
	{
		return 224;
	}
}

void question100::averagev_net(Mat img, int row, int col, int range)
{
	double allr = 0;
	double allg = 0;
	double allb = 0;
	for (int i = row * range; i < row * range + range; i++)
	{
		for (int j = col * range; j < col * range + range; j++)
		{
			allr += img.at<Vec3b>(i, j)[0];
			allg += img.at<Vec3b>(i, j)[1];
			allb += img.at<Vec3b>(i, j)[2];
		}
	}

	double averager = allr / (range * range);
	double averageg = allg / (range * range);
	double averageb = allb / (range * range);
	for (int i = row * range; i < row * range + range; i++)
	{
		for (int j = col * range; j < col * range + range; j++)
		{
			img.at<Vec3b>(i, j)[0] = averager;
			img.at<Vec3b>(i, j)[1] = averageg;
			img.at<Vec3b>(i, j)[2] = averageb;
		}
	}
}

void question100::max_net(Mat img, int row, int col, int range)
{
	double maxr = 0;
	double maxg = 0;
	double maxb = 0;
	for (int i = row * range; i < row * range + range; i++)
	{
		for (int j = col * range; j < col * range + range; j++)
		{
			double r = img.at<Vec3b>(i, j)[0];
			double g = img.at<Vec3b>(i, j)[1];
			double b = img.at<Vec3b>(i, j)[2];
			
			if (r > maxr)
			{
				maxr = r;
			}
			if (g > maxg)
			{
				maxg = g;
			}
			if (b > maxb)
			{
				maxb = b;
			}
		}
	}

	for (int i = row * range; i < row * range + range; i++)
	{
		for (int j = col * range; j < col * range + range; j++)
		{
			img.at<Vec3b>(i, j)[0] = maxr;
			img.at<Vec3b>(i, j)[1] = maxg;
			img.at<Vec3b>(i, j)[2] = maxb;
		}
	}
}

double question100::get_median(vector<double>& arr)
{
	//sort
	sort(arr.begin(),arr.end());
	//get
	int index = arr.size() / 2;
	if (index < 0 || index > arr.size())
	{
		return 0;
	}
	return arr[index];
}

std::vector<int> question100::vecmattrans(double mat[3][3], vector<int> vec)
{
	int size = vec.size();

	if (size != 3)
	{
		return vec;
	}
	vector<int> ret;
	for (int i = 0; i < 3; i++)
	{
		int v = 0;
		for (int j = 0; j < 3; j++)
		{
			v += vec[j] * mat[i][j];
		}
		ret.push_back(v);
	}
	return ret;
}

int question100::MatrixInvert(double m[3][3], double output[3][3])
{
	double vec[2], scale_sq, inv_sq_scale;
	int i, j;

	/*--------------------------------------------------------------------*\
	If the matrix is null, return the identity matrix
	\*--------------------------------------------------------------------*/
	if (m == NULL)
	{
		memcpy(NULL, output, 3*3* sizeof(double));
		return(1);
	}

	/*--------------------------------------------------------------------*\
	Obtain the matrix scale
	\*--------------------------------------------------------------------*/
	vec[0] = m[0][0];
	vec[1] = m[0][1];
	scale_sq = vec[0] * vec[0] + vec[1] * vec[1];

	/*--------------------------------------------------------------------*\
	Check whether there is an inverse, and if not, return 0
	\*--------------------------------------------------------------------*/
	if (scale_sq < (.000000001 * .000000001))
		return(0);

	/*--------------------------------------------------------------------*\
	Need the inverse scale squared
	\*--------------------------------------------------------------------*/
	inv_sq_scale = 1.0 / scale_sq;

	/*--------------------------------------------------------------------*\
	The orientation vectors
	\*--------------------------------------------------------------------*/
	for (j = 0; j<2; j++)
	{
		for (i = 0; i<2; i++)
			output[j][i] = m[i][j] * inv_sq_scale;
		output[j][2] = 0.0;
	}

	/*--------------------------------------------------------------------*\
	The shift vectors
	\*--------------------------------------------------------------------*/
	for (i = 0; i<2; i++)
	{
		output[2][i] = 0.0;
		for (j = 0; j<2; j++)
			output[2][i] -= m[i][j] * m[2][j] * inv_sq_scale;
	}
	output[2][2] = 1.0;

	return(1);
}

