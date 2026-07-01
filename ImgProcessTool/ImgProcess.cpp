#include "ImgProcess.h"
#include <QMessageBox>
#include <QDebug>

ImgProcess::ImgProcess()
{
}


ImgProcess::~ImgProcess()
{
}

QImage ImgProcess::enlarge(QImage src, int times)
{
	if (times == 0)
	{
		return src;
	}
	cv::Mat mat = QImage2cvMat(src);
	int w;
	int h;
	if (times > 0)
	{
		w = mat.cols * (times+1);
		h = mat.cols * (times+1);
	}
	else {
		w = mat.cols / abs(times-1);
		h = mat.cols / abs(times-1);
	}
	
	resize(mat, mat, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
	return cvMat2QImage(mat);
}

QImage ImgProcess::rodateByCenter(QImage src, int angle)
{
	if (angle == 0)
	{
		return src;
	}
	cv::Mat mat = QImage2cvMat(src);

	cv::Point2f center((mat.cols-1) / 2.0, (mat.rows-1) / 2.0);
	cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Mat rotatedImage;
	cv::warpAffine(mat, rotatedImage, rotationMatrix, mat.size());

	return cvMat2QImage(rotatedImage);
}

QImage ImgProcess::rotate(QImage src, int angle)
{
	if (angle == 0)
	{
		return src;
	}
	cv::Mat mat = QImage2cvMat(src);

	cv::Point2f center((mat.cols-1) / 2.0, (mat.rows-1) / 2.0);
	cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Mat rotatedImage;

	cv::Rect bbox = cv::RotatedRect(center, rotationMatrix.size(), angle).boundingRect();

	rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	cv::warpAffine(mat, rotatedImage, rotationMatrix, mat.size());

	return cvMat2QImage(rotatedImage);
}

QImage ImgProcess::flip(QImage src, int flip)
{
	cv::Mat mat = QImage2cvMat(src);

	cv::flip(mat, mat, flip);

	return cvMat2QImage(mat);
}

QImage ImgProcess::bin(QImage src, int threshold)
{
	cv::Mat mat = QImage2cvMat(src);
	if (mat.channels() != 1)
	{
		cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	}
	
	cv::threshold(mat, mat, threshold, 255, cv::THRESH_BINARY);
	return cvMat2QImage(mat);
}

QImage ImgProcess::gray(QImage src)
{
	cv::Mat mat = QImage2cvMat(src);
	if (mat.channels() != 1)
	{
		cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	}
	return cvMat2QImage(mat);
}

QImage ImgProcess::Reverse(QImage src)
{
	cv::Mat mat = QImage2cvMat(src);
	bitwise_xor(mat, cv::Scalar(255), mat);
	return cvMat2QImage(mat);
}

QImage ImgProcess::LogTrans(QImage src, int c)
{
	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);

	cv::Mat lookUpTable(1, 256, CV_8U);                                    // 查找表
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>((c / 100.0)*log(1 + i / 255.0)*255.0);      // pow()是幂次运算

	LUT(srcImg, lookUpTable, dstImg);                                   // LUT 

	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::Gamma(QImage src, int gamma)
{
	if (gamma < 0)
		return src;

	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);

	cv::Mat lookUpTable(1, 256, CV_8U);                                    // 查找表
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma / 100.0)*255.0);      // pow()是幂次运算

	LUT(srcImg, lookUpTable, dstImg);                                   // LUT 

	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::Histeq(QImage src)
{
	cv::Mat mat = QImage2cvMat(src);

	if (mat.channels() != 1)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	}

	cv::equalizeHist(mat, mat);
	return cvMat2QImage(mat);
}

QImage ImgProcess::Linear(QImage src, int alpha, int beta)
{
	cv::Mat mat = QImage2cvMat(src);

	mat.convertTo(mat, -1, alpha / 100.0, beta - 100);

	return cvMat2QImage(mat);
}

QImage ImgProcess::CircleDetect(QImage src, int minRad, int maxRad)
{
	cv::Mat mat = QImage2cvMat(src);

	if (mat.channels() != 1)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	}

	medianBlur(mat, mat, 5);
	
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mat, circles, cv::HOUGH_GRADIENT, 1, mat.rows / 4, 100, 30, minRad, maxRad);

	for (int i = 0; i < circles.size(); i++)
	{
		cv::Vec3i c = circles[i];
		cv::Point center = cv::Point(c[0], c[1]);
		cv::circle(mat, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);                    // 画圆
		int radius = c[2];
		cv::circle(mat, center, radius+2, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
	}
	return cvMat2QImage(mat);
}

QImage ImgProcess::LineDetect(QImage src)
{
	cv::Mat mat = QImage2cvMat(src);
	cv::Mat dst;
	cv::Canny(mat, dst, 50, 200, 3);
	if (dst.channels() != 1)
	{
		cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
	}

	std::vector<cv::Vec4i> linesP;
	cv::HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < linesP.size(); i++)
	{
		cv::Vec4i l = linesP[i];
		cv::line(dst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	}
	return cvMat2QImage(dst);
}

QImage ImgProcess::Normalize(QImage src, int kernelsize)
{
	cv::Mat mat = QImage2cvMat(src);
	cv::Mat dst;

	blur(mat, dst, cv::Size(kernelsize, kernelsize), cv::Point(-1, -1));

	return cvMat2QImage(dst);
}

QImage ImgProcess::Gaussian(QImage src, int kernelsize)
{
	cv::Mat mat = QImage2cvMat(src);
	cv::Mat dst;

	GaussianBlur(mat, dst, cv::Size(kernelsize, kernelsize), 0, 0);

	return cvMat2QImage(dst);
}

QImage ImgProcess::Median(QImage src, int kernelsize)
{
	cv::Mat mat = QImage2cvMat(src);
	cv::Mat dst;

	medianBlur(mat, dst, kernelsize);

	return cvMat2QImage(dst);
}

QImage ImgProcess::Sobel(QImage src, int kernelsize)
{
	cv::Mat srcImg, dstImg, src_gray;
	srcImg = QImage2cvMat(src);

	GaussianBlur(srcImg, srcImg, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);     // 高斯模糊
	if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, cv::COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;

	cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

	cv::Sobel(src_gray, grad_x, CV_16S, 1, 0, kernelsize, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(src_gray, grad_y, CV_16S, 0, 1, kernelsize, 1, 0, cv::BORDER_DEFAULT);

	cv::convertScaleAbs(grad_x, abs_grad_x);            // 缩放，计算绝对值，并将结果转换为8位
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImg);

	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::Laplacian(QImage src, int kernelsize)
{
	cv::Mat srcImg, dstImg, src_gray;
	srcImg = QImage2cvMat(src);

	cv::GaussianBlur(srcImg, srcImg, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);       // 高斯模糊

	if (srcImg.channels() != 1)
		cv::cvtColor(srcImg, src_gray, cv::COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;

	cv::Mat abs_dst;                                                    // 拉普拉斯二阶导数
	cv::Laplacian(src_gray, dstImg, CV_16S, kernelsize, 1, 0, cv::BORDER_DEFAULT);

	convertScaleAbs(dstImg, dstImg);                                  // 绝对值8位
	return cvMat2QImage(dstImg);
}

QImage ImgProcess::Canny(QImage src, int kernelsize, int lowthreshold, int highthreshold)
{
	cv::Mat srcImg, dstImg, src_gray, detected_edges;
	srcImg = QImage2cvMat(src);

	dstImg.create(srcImg.size(), srcImg.type());
	if (srcImg.channels() != 1)
		cv::cvtColor(srcImg, src_gray, cv::COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;
	blur(src_gray, detected_edges, cv::Size(3, 3));     // 平均滤波平滑
	cv::Canny(detected_edges, detected_edges, lowthreshold, highthreshold, kernelsize);
	dstImg = cv::Scalar::all(0);
	srcImg.copyTo(dstImg, detected_edges);

	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::Erode(QImage src, int elem, int kernel, int times)
{
	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);
	int erosion_type = 0;
	if (elem == 0)  {erosion_type = cv::MORPH_RECT; }
	else if (elem == 1) { erosion_type = cv::MORPH_CROSS; }
	else if (elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
	cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	cv::erode(srcImg, dstImg, element, cv::Point(-1, -1), times);
	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::Dilate(QImage src, int elem, int kernel, int times)
{
	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);
	int dilation_type = 0;
	if (elem == 0) { dilation_type = cv::MORPH_RECT; }
	else if (elem == 1) { dilation_type = cv::MORPH_CROSS; }
	else if (elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
	cv::Mat element = cv::getStructuringElement(dilation_type, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	cv::dilate(srcImg, dstImg, element, cv::Point(-1, -1), times);
	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::OpenOperation(QImage src, int elem, int kernel, int times)
{
	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);
	cv::Mat element = cv::getStructuringElement(elem, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	cv::morphologyEx(srcImg, dstImg, cv::MORPH_OPEN, element, cv::Point(-1, -1), times);
	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::CloseOperation(QImage src, int elem, int kernel, int times)
{
	cv::Mat srcImg, dstImg;
	srcImg = QImage2cvMat(src);
	cv::Mat element = getStructuringElement(elem, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	cv::morphologyEx(srcImg, dstImg, cv::MORPH_CLOSE, element, cv::Point(-1, -1), times);
	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::TopHat(QImage src, int elem, int kernel)
{
	cv::Mat srcImg, grayImg, dstImg;
	srcImg = QImage2cvMat(src);
	cv::Mat element = getStructuringElement(elem, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	if (srcImg.channels() != 1)
		cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	else
		grayImg = srcImg.clone();

	cv::morphologyEx(grayImg, dstImg, cv::MORPH_TOPHAT, element);
	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::BlackHat(QImage src, int elem, int kernel)
{
	cv::Mat srcImg, grayImg, dstImg;
	srcImg = QImage2cvMat(src);
	cv::Mat element = cv::getStructuringElement(elem, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));
	if (srcImg.channels() != 1)
		cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	else
		grayImg = srcImg.clone();
	cv::morphologyEx(grayImg, dstImg, cv::MORPH_BLACKHAT, element);
	QImage dst =cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::MorphologyGradient(QImage src, int elem, int kernel)
{
	cv::Mat srcImg, grayImg, dstImg;
	srcImg = QImage2cvMat(src);
	cv::Mat element = cv::getStructuringElement(elem, cv::Size(2 * kernel + 3, 2 * kernel + 3), cv::Point(kernel + 1, kernel + 1));

	if (srcImg.channels() != 1)
		cvtColor(srcImg, grayImg, cv::COLOR_BGR2GRAY);
	else
		grayImg = srcImg.clone();
	cv::morphologyEx(grayImg, dstImg, cv::MORPH_GRADIENT, element);

	QImage dst = cvMat2QImage(dstImg);
	return dst;
}

QImage ImgProcess::cvMat2QImage(const cv::Mat& mat)
{
	switch (mat.type())
	{
	case CV_8UC1:
	{
		QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
		return img.copy();
	}
	case CV_8UC3:
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
		QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return img.copy();
	}
	case CV_8UC4:
	{
		QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return img.copy();
	}
	default:
		qDebug() << "unsupported cv::Mat type: " << mat.type();
		return QImage();
	}
}

cv::Mat ImgProcess::QImage2cvMat(QImage img)
{
	cv::Mat mat;
	switch (img.format())
	{
	case QImage::Format_Grayscale8:
	{
		mat = cv::Mat(img.height(), img.width(), CV_8UC1,
			static_cast<uchar*>(img.bits()), img.bytesPerLine());
		break;
	}
	case QImage::Format_RGB888:
	{
		mat = cv::Mat(img.height(), img.width(), CV_8UC3,
			static_cast<uchar*>(img.bits()), img.bytesPerLine());
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
		break;
	}
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32:
	{
		mat = cv::Mat(img.height(), img.width(), CV_8UC4,
			static_cast<uchar*>(img.bits()), img.bytesPerLine());
//		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
		break;
	}
	default:
		qDebug() << "unsupported type: " << img.format();
		break;
	}
	return mat.clone();
}

QImage ImgProcess::splitBGR(QImage src, int color)
{
	cv::Mat mat = QImage2cvMat(src);
	std::vector<cv::Mat> channels;
	cv::split(mat, channels);
	if (channels.size() != 3)
	{
		QMessageBox::information(nullptr, "error", "this not a rgb img");
		return src;
	}
	return cvMat2QImage(channels[color]);
}

QImage ImgProcess::splitColor(QImage src, QString model, int color)
{
	cv::Mat img = QImage2cvMat(src);
	cv::Mat img_rgb, img_hsv, img_hls, img_yuv, img_dst;

	if (img.channels() == 1)
	{
		QMessageBox message(QMessageBox::Information, QString::fromLocal8Bit("提示"), QString::fromLocal8Bit("该图像为灰度图像。"));
		message.exec();
		return src;
	}
	else
	{
		std::vector <cv::Mat> vecRGB, vecHsv, vecHls, vecYuv;
		img_hsv.create(img.rows, img.cols, CV_8UC3);
		img_hls.create(img.rows, img.cols, CV_8UC3);

		cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
		cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
		cvtColor(img, img_hls, cv::COLOR_BGR2HLS);
		cvtColor(img, img_yuv, cv::COLOR_BGR2YUV);

		split(img_rgb, vecRGB);
		split(img_hsv, vecHsv);
		split(img_hls, vecHls);
		split(img_yuv, vecYuv);

		if (model == "RGB")
			img_dst = vecRGB[color];
		else if (model == "HSV")
			img_dst = vecHsv[color];
		else if (model == "HLS")
			img_dst = vecHls[color];
		else if (model == "YUV")
			img_dst = vecYuv[color];
		else
			img_dst = img;

		QImage dst = cvMat2QImage(img_dst);
		return dst;
	}
}
