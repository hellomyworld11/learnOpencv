#include "ImgProcessTool.h"
#include <QFormLayout>
#include <QPushButton>
#include <QGridLayout>

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
	//4.图像窗口
	InitImgView();
	//其它视图
	InitUiView();
	InitLayOut();
}

void ImgProcessTool::InitLayOut()
{
	//takeCentralWidget();
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
	pPainterView_ = new QDockWidget(QString::fromLocal8Bit("绘制"), this);
	pPainterView_->setFeatures(QDockWidget::DockWidgetClosable);
	
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


	QButtonGroup *btnGrp = new QButtonGroup();
	void(QButtonGroup::*ptr)(int) = static_cast<void(QButtonGroup::*)(int)>(&QButtonGroup::buttonClicked);
	connect(btnGrp, ptr, this, &ImgProcessTool::on_ToolButtonClicked);
	btnGrp->addButton(btn_pen, 1);
	btnGrp->addButton(btn_line, 2);
	btnGrp->addButton(btn_circle, 3);
	btnGrp->addButton(btn_ellipse, 4);
	btnGrp->addButton(btn_triangle, 5);
	btnGrp->addButton(btn_rhombus, 6);
	btnGrp->addButton(btn_rect, 7);
	btnGrp->addButton(btn_square, 8);
	btnGrp->addButton(btn_six, 9);

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

void ImgProcessTool::on_ToolButtonClicked(int id)
{

}
