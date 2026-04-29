#pragma once
#include <QImage>
#include <QVector>

class ImgProcess
{
public:
	ImgProcess();

	~ImgProcess();


private:
	QVector<QImage> imgs_;
};

