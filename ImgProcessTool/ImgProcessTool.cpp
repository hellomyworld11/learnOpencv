#include "ImgProcessTool.h"
#include <QFormLayout>
#include <QPushButton>
#include <QGridLayout>
#include <QDebug>

ImgProcessTool::ImgProcessTool(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	resize(800, 600);
	InitForm();
}

void ImgProcessTool::InitForm()
{
	//1.添加菜单 	
	pMenuBar_ = new MenuBar(this);
	setMenuBar(pMenuBar_);
	
	//其它视图
	InitUiView();

	//4.图像窗口
	InitImgView();

	InitLayOut();
}

void ImgProcessTool::InitLayOut()
{
	//takeCentralWidget();
	setDockNestingEnabled(true);

	setCentralWidget(pImgView_);
	addDockWidget(Qt::LeftDockWidgetArea, pPainterView_);
	addDockWidget(Qt::BottomDockWidgetArea, pOutPutView_);
	addDockWidget(Qt::RightDockWidgetArea, pPropertyView_);

	pPainterView_->setFixedWidth(100);
	pPropertyView_->setMaximumHeight(240);
	splitDockWidget(pPainterView_, pImgView_, Qt::Horizontal);
	tabifyDockWidget(pOutPutView_, pPropertyView_);
	pOutPutView_->raise();
}

void ImgProcessTool::InitImgView()
{
	pImgView_ = new QDockWidget(QString::fromLocal8Bit("图像显示"), this);
	//pImgView_->setAllowedAreas()
	pImgView_->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);
	
	imgwin_ = new ImgWindow(pImgView_);
	imgwin_->setScaledContents(true);
	QImage image = QImage(500, 500, QImage::Format_ARGB32);
	image.fill(qRgb(255, 255, 255));
	imgwin_->SetPenWidth(2);
	imgwin_->SetColor(Qt::black);
	imgwin_->SetImage(image);
	imgwin_->setPixmap(QPixmap::fromImage(image));
//	imgProcess_.Push(image);
	PushState();
	propertyInfo_.Update(image);

	//输出信息
	bool bRet = connect(imgwin_, &ImgWindow::mouseInfoChanged,
		this, [this](const QString &info) {
		Report(info);
	});
	if (!bRet) {
		qDebug() << "信号槽连接失败！";
	}
	else {
		qDebug() << "信号槽连接成功";
	}



	QAction *pUndo = pMenuBar_->GetAction(QString::fromLocal8Bit("撤销"));
	if (pUndo)
	{
		pUndo->setEnabled(false);
	}
	QAction *pRedo = pMenuBar_->GetAction(QString::fromLocal8Bit("重做"));
	if (pRedo)
	{
		pRedo->setEnabled(false);
	}

// 	iter = imgVector[0].end() - 1;						// 最后一个元素对应的迭代器指针

	// 增加滚动条
	ImgscrollArea = new QScrollArea();
	ImgscrollArea->setBackgroundRole(QPalette::Dark);
	ImgscrollArea->setAlignment(Qt::AlignCenter);
	ImgscrollArea->setWidget(imgwin_);
	pImgView_->setWidget(ImgscrollArea);
}

void ImgProcessTool::InitUiView()
{
	//1.输出
	pOutPutView_ = new QDockWidget(QString::fromLocal8Bit("输出"), this);
	pOutPutView_->setFeatures(QDockWidget::AllDockWidgetFeatures);
	pOutputEdit_ = new QTextEdit();
	pOutputEdit_->textCursor().movePosition(QTextCursor::End);
	pOutputEdit_->setWordWrapMode(QTextOption::NoWrap);
	pOutputEdit_->setReadOnly(true);
	pOutPutView_->setWidget(pOutputEdit_);

//	pOutPutView_->setAllowedAreas(Qt::DockWidgetArea::BottomDockWidgetArea);
	//3.绘制模块
	InitPainterView();
	
	//5.属性窗口
	InitPropertyView();
	
}

void ImgProcessTool::InitPainterView()
{
	pPainterView_ = new QDockWidget(QString::fromLocal8Bit("工具"), this);
	pPainterView_->setFeatures(QDockWidget::DockWidgetClosable |
		QDockWidget::DockWidgetFloatable);
	
	QPushButton *btn_pen = new QPushButton(QString::fromLocal8Bit("钢笔"), this);
	btn_pen->setFixedSize(35, 35);
	QPushButton *btn_line = new QPushButton(QString::fromLocal8Bit("线条"), this);
	btn_line->setFixedSize(35, 35);
	QPushButton *btn_circle = new QPushButton(QString::fromLocal8Bit("圆形"), this);
	btn_circle->setFixedSize(35, 35);
	QPushButton *btn_ellipse = new QPushButton(QString::fromLocal8Bit("椭圆"), this);
	btn_ellipse->setFixedSize(35, 35);
	QPushButton *btn_triangle = new QPushButton(QString::fromLocal8Bit("三角形"), this);
	btn_triangle->setFixedSize(35, 35);
	QPushButton *btn_rhombus = new QPushButton(QString::fromLocal8Bit("菱形"), this);
	btn_rhombus->setFixedSize(35, 35);
	QPushButton *btn_rect = new QPushButton(QString::fromLocal8Bit("长方形"), this);
	btn_rect->setFixedSize(35, 35);
	QPushButton *btn_square = new QPushButton(QString::fromLocal8Bit("正方形"), this);
	btn_square->setFixedSize(35, 35);
	QPushButton *btn_six = new QPushButton(QString::fromLocal8Bit("六边形"), this);
	btn_six->setFixedSize(35, 35);
	QPushButton *btn_pix = new QPushButton(QString::fromLocal8Bit("像素信息"), this);
	btn_pix->setFixedSize(35, 35);

	QButtonGroup *btnGrp = new QButtonGroup();
	void(QButtonGroup::*ptr)(int) = static_cast<void(QButtonGroup::*)(int)>(&QButtonGroup::buttonClicked);
	connect(btnGrp, ptr, this, &ImgProcessTool::on_ToolButtonClicked);
	btnGrp->addButton(btn_pen, (int)ToolBtnId::Pen);
	btnGrp->addButton(btn_line, (int)ToolBtnId::Line);
	btnGrp->addButton(btn_circle, (int)ToolBtnId::Circle);
	btnGrp->addButton(btn_ellipse, (int)ToolBtnId::Ellipse);
	btnGrp->addButton(btn_triangle, (int)ToolBtnId::Triangle);
	btnGrp->addButton(btn_rhombus, (int)ToolBtnId::Rhombus);
	btnGrp->addButton(btn_rect, (int)ToolBtnId::Rect);
	btnGrp->addButton(btn_square, (int)ToolBtnId::Square);
	btnGrp->addButton(btn_six, (int)ToolBtnId::Six);
	btnGrp->addButton(btn_pix, (int)ToolBtnId::Pix);

	QGridLayout *layout = new QGridLayout();
	layout->setAlignment(Qt::AlignTop);
	layout->addWidget(btn_pen, 0, 0);
	layout->addWidget(btn_line, 0, 1);
	layout->addWidget(btn_ellipse, 1, 0);
	layout->addWidget(btn_circle, 1, 1);
	layout->addWidget(btn_triangle, 2, 0);
	layout->addWidget(btn_rhombus, 2, 1);
	layout->addWidget(btn_rect, 3, 0);
	layout->addWidget(btn_square, 3, 1);
	layout->addWidget(btn_six, 4, 0);
	layout->addWidget(btn_pix, 4, 1);

	QWidget *painterwidget = new QWidget(pPainterView_);
	painterwidget->setLayout(layout);
	pPainterView_->setWidget(painterwidget);

}

void ImgProcessTool::InitPropertyView()
{
	pPropertyView_ = new QDockWidget(QString::fromLocal8Bit("属性"), this);
	pPropertyView_->setFeatures(QDockWidget::AllDockWidgetFeatures);

	propertyInfo_.pLineImg = new QLineEdit();
	propertyInfo_.pLineLen = new QLineEdit();
	propertyInfo_.pLineWid = new QLineEdit();
	propertyInfo_.pLineGray = new QLineEdit();
	propertyInfo_.pLineDepth = new QLineEdit();
	propertyInfo_.pLineImg->setReadOnly(true);
	propertyInfo_.pLineLen->setReadOnly(true);
	propertyInfo_.pLineWid->setReadOnly(true);
	propertyInfo_.pLineGray->setReadOnly(true);
	propertyInfo_.pLineDepth->setReadOnly(true);

	QFormLayout *pFormalayout = new QFormLayout();
	pFormalayout->addRow(QString::fromLocal8Bit("图像"), propertyInfo_.pLineImg);
	pFormalayout->addRow(QString::fromLocal8Bit("长度"), propertyInfo_.pLineLen);
	pFormalayout->addRow(QString::fromLocal8Bit("宽度"), propertyInfo_.pLineWid);
	pFormalayout->addRow(QString::fromLocal8Bit("深度"), propertyInfo_.pLineDepth);
	pFormalayout->addRow(QString::fromLocal8Bit("是否灰度图像"), propertyInfo_.pLineGray);

	QWidget *proertyWidget = new QWidget(pPropertyView_);
	proertyWidget->setLayout(pFormalayout);
	pPropertyView_->setWidget(proertyWidget);
}

void ImgProcessTool::RenderImage(QImage Img, bool bPushState)
{
	imgwin_->SetImage(Img);
	if (bPushState)
	{
		PushState();
	}
	imgwin_->setPixmap(QPixmap::fromImage(Img));
	imgwin_->resize(Img.width(), Img.height());
	propertyInfo_.Update(Img);
}

void ImgProcessTool::Report(QString str)
{
	pOutputEdit_->append(str);
}

void ImgProcessTool::PushState()
{
	imgHistory_.pushState(imgwin_->GetImg());
	pMenuBar_->UpdataUIState();
}

void ImgProcessTool::on_ToolButtonClicked(int id)
{
	qDebug() << "id: " << id;

	switch (id)
	{
	case (int)ToolBtnId::Pen:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Pen);
		Report(QString::fromLocal8Bit("钢笔"));
		break;
	case (int)ToolBtnId::Line:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Line);
		Report(QString::fromLocal8Bit("直线"));
		break;
	case (int)ToolBtnId::Circle:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Circle);
		Report(QString::fromLocal8Bit("圆形"));
		break;
	case (int)ToolBtnId::Ellipse:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Ellipse);
		Report(QString::fromLocal8Bit("椭圆"));
		break;
	case (int)ToolBtnId::Triangle:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Triangle);
		Report(QString::fromLocal8Bit("三角"));
		break;
	case (int)ToolBtnId::Rhombus:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Rhombus);
		Report(QString::fromLocal8Bit("菱形"));
		break;
	case (int)ToolBtnId::Rect:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Rect);
		Report(QString::fromLocal8Bit("长方形"));
		break;
	case (int)ToolBtnId::Square:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Sqaure);
		Report(QString::fromLocal8Bit("正方形"));
		break;
	case (int)ToolBtnId::Six:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Hexagon);
		Report(QString::fromLocal8Bit("六边形"));
		break;
	case (int)ToolBtnId::Pix:
		imgwin_->SetShape(ImgWindow::ToolOrShape::Pix);
		Report(QString::fromLocal8Bit("像素信息获取"));
		break;
	default:
		break;
	}
}

bool ImgProcessTool::PropertyControl::Update(QImage& img)
{
	pLineLen->setText(QString::number(img.width()));
	pLineWid->setText(QString::number(img.height()));
	pLineDepth->setText(QString::number(img.depth()));
	pLineImg->setText("img");
	if (img.depth() == 8 || img.depth() == 1 )
	{
		pLineGray->setText(QString::fromLocal8Bit("是") );
	}
	else {
		pLineGray->setText(QString::fromLocal8Bit("否"));
	}
	return true;
}
