#include "ImgWindow.h"
#include <QtGui>
#include <QtWidgets>

ImgWindow::ImgWindow(QWidget *parent)
	: QLabel(parent)
{
	ui.setupUi(this);
}

ImgWindow::~ImgWindow()
{
}

void ImgWindow::Paint(QImage &img)
{
	QPainter p(&img);
	QPen apen;
	apen.setWidth(width_);
	apen.setColor(color_);
	p.setPen(apen);
	p.setRenderHint(QPainter::RenderHint::Antialiasing, true);

	int x1, y1, x2, y2;
	x1 = lastPoint_.x();
	y1 = lastPoint_.y();
	x2 = endPoint_.x();
	y2 = endPoint_.y();

	switch (type_)
	{
	case ToolOrShape::Pen:
	{
		p.drawLine(lastPoint_, endPoint_);
		lastPoint_ = endPoint_;
		break;
	}
	case ToolOrShape::Line:
	{
		p.drawLine(lastPoint_, endPoint_);
		break;
	}
	case ToolOrShape::Circle:
	{
		double drad = QLineF(QPointF(x1, y1), QPointF(x2, y2)).length();
		p.drawEllipse(QPointF(x1, y1), drad, drad);
		break;
	}
	case ToolOrShape::Ellipse:
	{
		p.drawEllipse(x1, y1, x2 - x1, y2 - y1);
		break;
	}
	case ToolOrShape::Rhombus:
	{
		int top = (y1 < y2) ? y1 : y2;
		int bottom = (y1 > y2) ? y1 : y2;
		int left = (x1 < x2) ? x1 : x2;
		int right = (x1 > x2) ? x1 : x2;
		QPoint points[4] = {
			QPoint(left, (top + bottom) / 2),
			QPoint((left + right) / 2, bottom),
			QPoint(right, (top + bottom) / 2),
			QPoint((left + right) / 2, top)
		};
		p.drawPolygon(points, 4);
		break;
	}
	case ToolOrShape::Triangle:
	{
		int top = (y1<y2) ? y1 : y2;
		int bottom = (y1 > y2) ? y1 : y2;
		int left = (x1 < x2) ? x1 : x2;
		int right = (x1 > x2) ? x1 : x2;

		QPoint points[3] = {QPoint(left, bottom), QPoint(right, bottom), QPoint((left+right)/2, top)};
		p.drawPolygon(points, 3);
		break;
	}
	case ToolOrShape::Rect:
	{
		p.drawRect(x1, y1, (x2 - x1), (y2 - y1));
		break;
	}
	case ToolOrShape::Sqaure:
	{
		double len = (x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1); 
		p.drawRect(x1, y1, len, len);
		break;
	}
	case ToolOrShape::Hexagon:
	{
		break;
	}
	default:
		break;
	}
	update();
}

void ImgWindow::SetImage(QImage img, QString strPath)
{
	img_ = img;
	tmpImg_ = img;
	imgpath_ = strPath;
}

void ImgWindow::SetShape(ToolOrShape type)
{
	type_ = type;
	if (type_ == ToolOrShape::Pix)
	{
		setMouseTracking(true);
	}
	else {
		setMouseTracking(false);
	}
}

void ImgWindow::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	if (isDrawing_)
	{
		painter.drawImage(0, 0, tmpImg_);
	}
	else {
		painter.drawImage(0, 0, img_);
	}
}

void ImgWindow::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton)
	{
		lastPoint_ = event->pos();
		isDrawing_ = true;
	}
}

void ImgWindow::mouseMoveEvent(QMouseEvent *event)
{
	if (type_ == ToolOrShape::Pix)
	{
		int x = event->pos().x();
		int y = event->pos().y();
		QRgb rgb = img_.pixel(x, y);
		QString str = QString("Pos: (%1, %2) RGB(%3, %4, %5)").arg(x).arg(y).arg(qRed(rgb)).arg(qGreen(rgb)).arg(qBlue(rgb));
		qDebug() << "准备发射信号mouseInfoChanged:" << str;  // ← 添加调试
		emit mouseInfoChanged(str);
	}
	if (event->buttons() & Qt::LeftButton)
	{
		endPoint_ = event->pos();
		tmpImg_ = img_;
		if (type_ == ToolOrShape::Pen)
		{
			Paint(img_);
		}
		else 
		{
			Paint(tmpImg_);
		}
	}
}

void ImgWindow::mouseReleaseEvent(QMouseEvent *)
{
	isDrawing_ = false;
	if (type_ != ToolOrShape::Pen)
	{
		Paint(img_);
	}
}
