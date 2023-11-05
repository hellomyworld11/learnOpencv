#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class question100
{
public:
	question100();
	~question100();

public:
	//1. tutorial
	void tutorial_que();
	void testMain();
private:
	//1. que1: rgb -> bgr Í¨µÀ  ¡î
	Mat change_channel(Mat img);
	//2. GrayScale				¡î
	Mat gray_scale(Mat img);
	//3. Thresholding			¡î
	Mat thresholding(Mat img);
	//4. otu's method			¡î
	Mat otus_ts(Mat img);
	//5. HSV translation		¡î
	Mat hdv_trans(Mat img);
	//6. reduce					¡î¡î
	Mat reduce(Mat img);
	//7. average Pooling		¡î¡î
	Mat average_pooling(Mat img);
	//8. max pooling            ¡î¡î
	Mat max_pooling(Mat img);
	//9. Gaussian Filter        ¡î¡î¡î¡î
	Mat gaussian_filter(Mat img);
	//10. median  Filter
	Mat median_filter(Mat img);
	//11. average Filter
	Mat average_filter(Mat img);
	//12. motion Filter
	Mat motion_filter(Mat img);
	//...
	//20. Histogram
	Mat histogram(Mat img);
	//24. Gamma Correction
	Mat gamma_correction(Mat img, double gamma_c, double gamma_g);
	//25. Nearest-neighbor Interpolation
	Mat nearest_neiInterpolation(Mat img);
	//...
	//28. Afine Transformations  move
	Mat afine_transformations_move(Mat img);
	//29. Afine Transformations  scale
	Mat afine_transformations_scale(Mat img);
private:
	// max of three double
	double maxdouble(double a, double b, double c);
	double mindouble(double a, double b, double c);
	double get_hue(double dmin, double dmax, double r, double g, double b);
	void rgb_to_hsv(double rgb[3], double hsv[3]);
	void hsv_to_rgb(double hsv[3], double rgb[3]);
	double quantify(double value);
	void averagev_net(Mat img, int row, int col, int range);
	void max_net(Mat img, int row, int col, int range);
	double get_median(vector<double>& arr);
	vector<int> vecmattrans(double mat[3][3], vector<int> vec);
	int MatrixInvert(
		double m[3][3],
		double output[3][3]);
};

