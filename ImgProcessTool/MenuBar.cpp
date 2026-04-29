#include "MenuBar.h"
#include <QMenu>
#include <QAction>
#include "ImgProcessTool.h"
#include <QDebug>


MenuBar::MenuBar(QWidget *parent)
	: QMenuBar(parent)
{
	pParent_ = dynamic_cast<ImgProcessTool*>(parent);
	Q_ASSERT(pParent_);
	Init();
}

MenuBar::~MenuBar()
{
}

void MenuBar::Init()
{
	Menu_File();
	Menu_Edit();
	Menu_View();

	// 独立顶级菜单
	Menu_GeoTransform();
	Menu_GrayTransform();
	Menu_ImageEnhance();
	Menu_ColorModel();

	Menu_Morphology();
	Menu_Help();
}

void MenuBar::Menu_File()
{
	QMenu *pFile = addMenu(QString::fromLocal8Bit("文件"));
	QAction *pNewFile = new QAction(QIcon(), QString::fromLocal8Bit("新建"), this);
	connect(pNewFile, &QAction::triggered, this, &MenuBar::NewFile);
	
	QAction *pOpenFile = new QAction(QIcon(), QString::fromLocal8Bit("打开"), this);
	connect(pOpenFile, &QAction::triggered, this, &MenuBar::OpenFile);

	QAction *pSaveFile = new QAction(QIcon(), QString::fromLocal8Bit("保存"), this);
	connect(pSaveFile, &QAction::triggered, this, &MenuBar::SaveFile);

	QAction *pSaveAs = new QAction(QIcon(), QString::fromLocal8Bit("另存为"), this);
	connect(pSaveAs, &QAction::triggered, this, &MenuBar::SaveAs);


	pFile->addAction(pNewFile);
	pFile->addAction(pOpenFile);
	pFile->addAction(pSaveFile);
	pFile->addAction(pSaveAs);
}

// ====================== 编辑菜单 ======================
void MenuBar::Menu_Edit()
{
	QMenu *pEdit = addMenu(QString::fromLocal8Bit("编辑"));

	QAction *pUndo = new QAction(QIcon(), QString::fromLocal8Bit("撤销"), this);
	connect(pUndo, &QAction::triggered, this, &MenuBar::Undo);

	QAction *pRedo = new QAction(QIcon(), QString::fromLocal8Bit("重做"), this);
	connect(pRedo, &QAction::triggered, this, &MenuBar::Redo);

	QAction *pFullScreen = new QAction(QIcon(), QString::fromLocal8Bit("全屏显示"), this);
	connect(pFullScreen, &QAction::triggered, this, &MenuBar::FullScreen);

	QAction *pExitFullScreen = new QAction(QIcon(), QString::fromLocal8Bit("退出全屏"), this);
	connect(pExitFullScreen, &QAction::triggered, this, &MenuBar::ExitFullScreen);

	QAction *pFind = new QAction(QIcon(), QString::fromLocal8Bit("查找"), this);
	connect(pFind, &QAction::triggered, this, &MenuBar::Find);

	pEdit->addAction(pUndo);
	pEdit->addAction(pRedo);
	pEdit->addSeparator();
	pEdit->addAction(pFullScreen);
	pEdit->addAction(pExitFullScreen);
	pEdit->addSeparator();
	pEdit->addAction(pFind);
}

// ====================== 视图菜单 ======================
void MenuBar::Menu_View()
{
	QMenu *pView = addMenu(QString::fromLocal8Bit("视图"));

	QAction *pToolBox = new QAction(QIcon(), QString::fromLocal8Bit("工具箱"), this);
	connect(pToolBox, &QAction::triggered, this, &MenuBar::ToolBox);

	QAction *pGeo = new QAction(QIcon(), QString::fromLocal8Bit("几何变换"), this);
	connect(pGeo, &QAction::triggered, this, &MenuBar::GeoTranslate);

	QAction *pGray = new QAction(QIcon(), QString::fromLocal8Bit("灰度变换"), this);
	connect(pGray, &QAction::triggered, this, &MenuBar::GrayTranslate);

	QAction *pEnhance = new QAction(QIcon(), QString::fromLocal8Bit("图像增强"), this);
	connect(pEnhance, &QAction::triggered, this, &MenuBar::EnhanceSmooth);

	QAction *pMorph = new QAction(QIcon(), QString::fromLocal8Bit("形态学处理"), this);
	connect(pMorph, &QAction::triggered, this, &MenuBar::MorphologyProcess);

	QAction *pColor = new QAction(QIcon(), QString::fromLocal8Bit("颜色模型"), this);
	connect(pColor, &QAction::triggered, this, &MenuBar::ColorModel);

	QAction *pImageWin = new QAction(QIcon(), QString::fromLocal8Bit("图像窗口"), this);
	connect(pImageWin, &QAction::triggered, this, &MenuBar::ImageWindow);

	QAction *pOutputWin = new QAction(QIcon(), QString::fromLocal8Bit("输出窗口"), this);
	connect(pOutputWin, &QAction::triggered, this, &MenuBar::OutputWindow);

	QAction *pPropertyWin = new QAction(QIcon(), QString::fromLocal8Bit("属性窗口"), this);
	connect(pPropertyWin, &QAction::triggered, this, &MenuBar::PropertyWindow);

	QAction *pFileBar = new QAction(QIcon(), QString::fromLocal8Bit("文件工具栏"), this);
	connect(pFileBar, &QAction::triggered, this, &MenuBar::FileToolBar);

	QAction *pDrawBar = new QAction(QIcon(), QString::fromLocal8Bit("绘图工具栏"), this);
	connect(pDrawBar, &QAction::triggered, this, &MenuBar::DrawToolBar);

	pView->addAction(pToolBox);
	pView->addSeparator();
	pView->addAction(pGeo);
	pView->addAction(pGray);
	pView->addAction(pEnhance);
	pView->addAction(pMorph);
	pView->addAction(pColor);
	pView->addSeparator();
	pView->addAction(pImageWin);
	pView->addAction(pOutputWin);
	pView->addAction(pPropertyWin);
	pView->addSeparator();
	pView->addAction(pFileBar);
	pView->addAction(pDrawBar);
}

// ====================== 几何变换（顶级菜单） ======================
void MenuBar::Menu_GeoTransform()
{
	QMenu *pMenu = addMenu(QString::fromLocal8Bit("几何变换"));

	QAction *pAutoSize = new QAction(QIcon(), QString::fromLocal8Bit("自适应"), this);
	QAction *pLarge = new QAction(QIcon(), QString::fromLocal8Bit("放大"), this);
	QAction *pSmall = new QAction(QIcon(), QString::fromLocal8Bit("缩小"), this);
	connect(pAutoSize, &QAction::triggered, this, &MenuBar::AutoSize);
	connect(pLarge, &QAction::triggered, this, &MenuBar::Large);
	connect(pSmall, &QAction::triggered, this, &MenuBar::Small);

	QMenu *pMenuImgSize = new QMenu(QString::fromLocal8Bit("图像缩放"));
	pMenuImgSize->addAction(pAutoSize);
	pMenuImgSize->addAction(pLarge);
	pMenuImgSize->addAction(pSmall);

	QAction *pRotate = new QAction(QIcon(), QString::fromLocal8Bit("顺时针旋转"), this);
	QAction *pRRotate = new QAction(QIcon(), QString::fromLocal8Bit("逆时针旋转"), this);
	QAction *pCRotate = new QAction(QIcon(), QString::fromLocal8Bit("中心旋转"), this);
	connect(pRotate, &QAction::triggered, this, &MenuBar::Rotate);
	connect(pRRotate, &QAction::triggered, this, &MenuBar::RRotate);
	connect(pCRotate, &QAction::triggered, this, &MenuBar::CRotate);

	QMenu *pMenuImgRotate = new QMenu(QString::fromLocal8Bit("图像旋转"));
	pMenuImgRotate->addAction(pRotate);
	pMenuImgRotate->addAction(pRRotate);
	pMenuImgRotate->addAction(pCRotate);

	QAction *pHFlip = new QAction(QIcon(), QString::fromLocal8Bit("水平翻转"), this);
	QAction *pVFlip = new QAction(QIcon(), QString::fromLocal8Bit("垂直翻转"), this);
	connect(pHFlip, &QAction::triggered, this, &MenuBar::HFlip);
	connect(pVFlip, &QAction::triggered, this, &MenuBar::VFlip);

	QMenu *pMenuImgFlip = new QMenu(QString::fromLocal8Bit("图像翻转"));
	pMenuImgFlip->addAction(pHFlip);
	pMenuImgFlip->addAction(pVFlip);

	pMenu->addMenu(pMenuImgSize);
	pMenu->addMenu(pMenuImgRotate);
	pMenu->addMenu(pMenuImgFlip);
}

// ====================== 灰度变换（顶级菜单） ======================
void MenuBar::Menu_GrayTransform()
{
	QMenu *pMenu = addMenu(QString::fromLocal8Bit("灰度变换"));

	QAction *pLogtrans = new QAction(QIcon(), QString::fromLocal8Bit("对数变换"), this);
	QAction *pGamma = new QAction(QIcon(), QString::fromLocal8Bit("伽马变换"), this);
	QAction *pHisteq = new QAction(QIcon(), QString::fromLocal8Bit("直方图均衡化"), this);
	connect(pLogtrans, &QAction::triggered, this, &MenuBar::AutoSize);
	connect(pGamma, &QAction::triggered, this, &MenuBar::Large);
	connect(pHisteq, &QAction::triggered, this, &MenuBar::Small);

	QMenu *pMenuNolinear = new QMenu(QString::fromLocal8Bit("非线性变换"));
	pMenuNolinear->addAction(pLogtrans);
	pMenuNolinear->addAction(pGamma);
	pMenuNolinear->addAction(pHisteq);

	QAction *pBin = new QAction(QIcon(), QString::fromLocal8Bit("二值图"), this);
	QAction *pGray = new QAction(QIcon(), QString::fromLocal8Bit("灰度图"), this);
	QAction *pReverse = new QAction(QIcon(), QString::fromLocal8Bit("反转变换"), this);
	QAction *pLinear = new QAction(QIcon(), QString::fromLocal8Bit("线性变换"), this);
	connect(pBin, &QAction::triggered, this, &MenuBar::AutoSize);
	connect(pGray, &QAction::triggered, this, &MenuBar::Large);
	connect(pReverse, &QAction::triggered, this, &MenuBar::Small);
	connect(pLinear, &QAction::triggered, this, &MenuBar::Small);

	pMenu->addMenu(pMenuNolinear);
	pMenu->addAction(pBin);
	pMenu->addAction(pGray);
	pMenu->addAction(pReverse);
	pMenu->addAction(pLinear);
}

// ====================== 图像增强（顶级菜单） ======================
void MenuBar::Menu_ImageEnhance()
{
	QMenu *pMenu = addMenu(QString::fromLocal8Bit("图像增强"));

	QAction *pCircle = new QAction(QIcon(), QString::fromLocal8Bit("圆检测"), this);
	QAction *pLine = new QAction(QIcon(), QString::fromLocal8Bit("直线检测"), this);
	connect(pCircle, &QAction::triggered, this, &MenuBar::CircleDetect);
	connect(pLine, &QAction::triggered, this, &MenuBar::LineDetect);

	QMenu *pMenuDetect = new QMenu(QString::fromLocal8Bit("Hough算子"));

	pMenuDetect->addAction(pCircle);
	pMenuDetect->addAction(pLine);
	
	QAction *pNormalize = new QAction(QIcon(), QString::fromLocal8Bit("简单滤波"), this);
	QAction *pGaussian = new QAction(QIcon(), QString::fromLocal8Bit("高斯滤波"), this);
	QAction *pMedian = new QAction(QIcon(), QString::fromLocal8Bit("中值滤波"), this);
	connect(pNormalize, &QAction::triggered, this, &MenuBar::Normalize);
	connect(pGaussian, &QAction::triggered, this, &MenuBar::Gaussian);
	connect(pMedian, &QAction::triggered, this, &MenuBar::Median);

	QAction *pSobel = new QAction(QIcon(), QString::fromLocal8Bit("Sobel算子"), this);
	QAction *pLaplacian = new QAction(QIcon(), QString::fromLocal8Bit("Laplacian算子"), this);
	QAction *pCanny = new QAction(QIcon(), QString::fromLocal8Bit("Canny算子"), this);
	connect(pSobel, &QAction::triggered, this, &MenuBar::Sobel);
	connect(pLaplacian, &QAction::triggered, this, &MenuBar::Gaussian);
	connect(pCanny, &QAction::triggered, this, &MenuBar::Canny);

	QMenu *pMenuSmooth = new QMenu(QString::fromLocal8Bit("平滑"));
	QMenu *pMenuSharpen = new QMenu(QString::fromLocal8Bit("锐化"));

	pMenuSmooth->addAction(pNormalize);
	pMenuSmooth->addAction(pGaussian);
	pMenuSmooth->addAction(pMedian);

	pMenuSharpen->addAction(pSobel);
	pMenuSharpen->addAction(pLaplacian);
	pMenuSharpen->addAction(pCanny);
	pMenuSharpen->addMenu(pMenuDetect);

	pMenu->addMenu(pMenuSmooth);
	pMenu->addMenu(pMenuSharpen);
}

// ====================== 颜色模型（顶级菜单） ======================
void MenuBar::Menu_ColorModel()
{
	QMenu *pMenu = addMenu(QString::fromLocal8Bit("颜色模型"));

	QAction *pRGB_R = new QAction(QIcon(), QString::fromLocal8Bit("R分量图"), this);
	QAction *pRGB_G = new QAction(QIcon(), QString::fromLocal8Bit("G分量图"), this);
	QAction *pRGB_B = new QAction(QIcon(), QString::fromLocal8Bit("B分量图"), this);
	connect(pRGB_R, &QAction::triggered, this, &MenuBar::RGB_R);
	connect(pRGB_G, &QAction::triggered, this, &MenuBar::RGB_G);
	connect(pRGB_B, &QAction::triggered, this, &MenuBar::RGB_B);

	QMenu *pMenuRGB = new QMenu(QString::fromLocal8Bit("RGB模型"));
	pMenuRGB->addAction(pRGB_R);
	pMenuRGB->addAction(pRGB_G);
	pMenuRGB->addAction(pRGB_B);

	QAction *pHSV_H = new QAction(QIcon(), QString::fromLocal8Bit("H分量图"), this);
	QAction *pHSV_S = new QAction(QIcon(), QString::fromLocal8Bit("S分量图"), this);
	QAction *pHSV_V = new QAction(QIcon(), QString::fromLocal8Bit("V分量图"), this);
	connect(pHSV_H, &QAction::triggered, this, &MenuBar::HSV_H);
	connect(pHSV_S, &QAction::triggered, this, &MenuBar::HSV_S);
	connect(pHSV_V, &QAction::triggered, this, &MenuBar::HSV_V);

	QMenu *pMenupHSV = new QMenu(QString::fromLocal8Bit("HSV模型"));
	pMenupHSV->addAction(pHSV_H);
	pMenupHSV->addAction(pHSV_S);
	pMenupHSV->addAction(pHSV_V);

	QAction *pYUV_Y = new QAction(QIcon(), QString::fromLocal8Bit("Y分量图"), this);
	QAction *pYUV_U = new QAction(QIcon(), QString::fromLocal8Bit("U分量图"), this);
	QAction *pYUV_V = new QAction(QIcon(), QString::fromLocal8Bit("V分量图"), this);
	connect(pYUV_Y, &QAction::triggered, this, &MenuBar::YUV_Y);
	connect(pYUV_U, &QAction::triggered, this, &MenuBar::YUV_U);
	connect(pYUV_V, &QAction::triggered, this, &MenuBar::YUV_V);

	QMenu *pMenupYUV = new QMenu(QString::fromLocal8Bit("YUV模型"));
	pMenupYUV->addAction(pYUV_Y);
	pMenupYUV->addAction(pYUV_U);
	pMenupYUV->addAction(pYUV_V);


	QAction *pHLS_H = new QAction(QIcon(), QString::fromLocal8Bit("H分量图"), this);
	QAction *pHLS_L = new QAction(QIcon(), QString::fromLocal8Bit("L分量图"), this);
	QAction *pHLS_S = new QAction(QIcon(), QString::fromLocal8Bit("S分量图"), this);
	connect(pHLS_H, &QAction::triggered, this, &MenuBar::HLS_H);
	connect(pHLS_L, &QAction::triggered, this, &MenuBar::HLS_L);
	connect(pHLS_S, &QAction::triggered, this, &MenuBar::HLS_S);

	QMenu *pMenupHLS = new QMenu(QString::fromLocal8Bit("HLS模型"));
	pMenupHLS->addAction(pHLS_H);
	pMenupHLS->addAction(pHLS_L);
	pMenupHLS->addAction(pHLS_S);

	pMenu->addMenu(pMenuRGB);
	pMenu->addMenu(pMenupHSV);
	pMenu->addMenu(pMenupYUV);
	pMenu->addMenu(pMenupHLS);
}

// ====================== 形态学菜单 ======================
void MenuBar::Menu_Morphology()
{
	QMenu *pMorph = addMenu(QString::fromLocal8Bit("形态学"));

	QAction *pErode = new QAction(QIcon(), QString::fromLocal8Bit("腐蚀"), this);
	connect(pErode, &QAction::triggered, this, &MenuBar::Erode);

	QAction *pDilate = new QAction(QIcon(), QString::fromLocal8Bit("膨胀"), this);
	connect(pDilate, &QAction::triggered, this, &MenuBar::Dilate);

	QAction *pOpen = new QAction(QIcon(), QString::fromLocal8Bit("开运算"), this);
	connect(pOpen, &QAction::triggered, this, &MenuBar::OpenOperation);

	QAction *pClose = new QAction(QIcon(), QString::fromLocal8Bit("闭运算"), this);
	connect(pClose, &QAction::triggered, this, &MenuBar::CloseOperation);

	QAction *pTopHat = new QAction(QIcon(), QString::fromLocal8Bit("顶帽操作"), this);
	connect(pTopHat, &QAction::triggered, this, &MenuBar::TopHat);

	QAction *pBlackHat = new QAction(QIcon(), QString::fromLocal8Bit("黑帽操作"), this);
	connect(pBlackHat, &QAction::triggered, this, &MenuBar::BlackHat);

	QAction *pGradient = new QAction(QIcon(), QString::fromLocal8Bit("形态学梯度"), this);
	connect(pGradient, &QAction::triggered, this, &MenuBar::MorphologyGradient);

	pMorph->addAction(pErode);
	pMorph->addAction(pDilate);
	pMorph->addSeparator();
	pMorph->addAction(pOpen);
	pMorph->addAction(pClose);
	pMorph->addSeparator();
	pMorph->addAction(pTopHat);
	pMorph->addAction(pBlackHat);
	pMorph->addAction(pGradient);
}

// ====================== 帮助菜单 ======================
void MenuBar::Menu_Help()
{
	QMenu *pHelp = addMenu(QString::fromLocal8Bit("帮助"));

	QAction *pAbout = new QAction(QIcon(), QString::fromLocal8Bit("关于"), this);
	connect(pAbout, &QAction::triggered, this, &MenuBar::About);

	QAction *pSet = new QAction(QIcon(), QString::fromLocal8Bit("设置"), this);
	connect(pSet, &QAction::triggered, this, &MenuBar::Settings);

	pHelp->addAction(pAbout);
	pHelp->addAction(pSet);
}

void MenuBar::NewFile()
{
	qDebug() << "NewFile...";
}

void MenuBar::OpenFile()
{

}

void MenuBar::SaveFile()
{

}

void MenuBar::SaveAs()
{

}

void MenuBar::Undo()
{
	qDebug() << "Undo...";
}

void MenuBar::Redo()
{
	qDebug() << "Redo...";
}

void MenuBar::FullScreen()
{
	qDebug() << "FullScreen...";
}

void MenuBar::ExitFullScreen()
{
	qDebug() << "ExitFullScreen...";
}

void MenuBar::Find()
{
	qDebug() << "Find...";
}

void MenuBar::ToolBox() { qDebug() << "ToolBox..."; }
void MenuBar::ImageWindow() { qDebug() << "ImageWindow..."; }
void MenuBar::OutputWindow() { qDebug() << "OutputWindow..."; }
void MenuBar::PropertyWindow() { qDebug() << "PropertyWindow..."; }
void MenuBar::FileToolBar() { qDebug() << "FileToolBar..."; }
void MenuBar::DrawToolBar() { qDebug() << "DrawToolBar..."; }

void MenuBar::MorphologyProcess()
{

}

void MenuBar::GeoTranslate() { qDebug() << "GeoTranslate"; }


void MenuBar::GrayTranslate()
{

}

void MenuBar::AutoSize()
{

}

void MenuBar::Large()
{

}

void MenuBar::Small()
{

}

void MenuBar::Rotate()
{

}

void MenuBar::RRotate()
{

}

void MenuBar::CRotate()
{

}

void MenuBar::HFlip()
{

}

void MenuBar::VFlip()
{

}

void MenuBar::Bin()
{

}

void MenuBar::Gray()
{

}

void MenuBar::Reverse()
{

}

void MenuBar::LogTrans()
{

}

void MenuBar::Gamma()
{

}

void MenuBar::Histeq()
{

}

void MenuBar::Linear()
{

}


void MenuBar::CircleDetect()
{

}

void MenuBar::LineDetect()
{

}

void MenuBar::EnhanceSmooth() { qDebug() << "EnhanceSmooth"; }


void MenuBar::ColorModel()
{

}

void MenuBar::Normalize()
{

}

void MenuBar::Gaussian()
{

}

void MenuBar::Median()
{

}

void MenuBar::Sobel()
{

}

void MenuBar::Laplacian()
{

}

void MenuBar::Canny()
{

}



void MenuBar::RGB_R()
{

}

void MenuBar::RGB_B()
{

}

void MenuBar::RGB_G()
{

}

void MenuBar::HSV_H()
{

}

void MenuBar::HSV_S()
{

}

void MenuBar::HSV_V()
{

}

void MenuBar::YUV_Y()
{

}

void MenuBar::YUV_U()
{

}

void MenuBar::YUV_V()
{

}

void MenuBar::HLS_H()
{

}

void MenuBar::HLS_L()
{

}

void MenuBar::HLS_S()
{

}

void MenuBar::Erode() { qDebug() << "Erode..."; }
void MenuBar::Dilate() { qDebug() << "Dilate..."; }
void MenuBar::OpenOperation() { qDebug() << "OpenOperation..."; }
void MenuBar::CloseOperation() { qDebug() << "CloseOperation..."; }
void MenuBar::TopHat() { qDebug() << "TopHat..."; }
void MenuBar::BlackHat() { qDebug() << "BlackHat..."; }
void MenuBar::MorphologyGradient() { qDebug() << "MorphologyGradient..."; }

void MenuBar::About() { qDebug() << "About..."; }
void MenuBar::Settings() { qDebug() << "Settings..."; }