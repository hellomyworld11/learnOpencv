#pragma once

#include <QWidget>
#include <QLabel>
#include "ui_ImgWindow.h"

class ImgWindow : public QLabel
{
	Q_OBJECT

public:
	ImgWindow(QWidget *parent = Q_NULLPTR);
	~ImgWindow();

	enum class ToolOrShape {
		None, Pen=1, Line, Ellipse, Circle, Triangle, Rhombus,	
		Rect, Sqaure, Hexagon, Pix
	};

	void Paint(QImage &img);

	QImage GetImg() { return img_; }

	QString GetPath() { return imgpath_; }

	void SetImage(QImage img, QString strPath = "");

	void SetShape(ToolOrShape type);

	void SetPenWidth(double width) { width_ = width; }

	void SetColor(QColor color) { color_ = color; }
protected:
	void paintEvent(QPaintEvent *);
	void mousePressEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);

signals:
	//像素信息传给主窗口
	void mouseInfoChanged(const QString &info);

private:
	Ui::ImgWindow ui;
	QImage img_;
	QImage tmpImg_;
	ToolOrShape type_;
	int width_;
	QColor color_;
	bool isDrawing_;
	QPoint lastPoint_;
	QPoint endPoint_;
	QString imgpath_;
};
