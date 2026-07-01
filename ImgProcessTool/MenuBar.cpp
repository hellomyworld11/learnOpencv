#include "MenuBar.h"
#include <QMenu>
#include <QAction>
#include "ImgProcessTool.h"
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include "HistoryMgr.h"
#include "SkinManager.h"


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

	actionsMap_[QString::fromLocal8Bit("撤销")] = pUndo;

	QAction *pRedo = new QAction(QIcon(), QString::fromLocal8Bit("重做"), this);
	connect(pRedo, &QAction::triggered, this, &MenuBar::Redo);

	actionsMap_[QString::fromLocal8Bit("重做")] = pRedo;

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
	pToolBox->setCheckable(true);
	pToolBox->setChecked(true);
	actionsMap_[QString::fromLocal8Bit("工具箱")] = pToolBox;


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
	connect(pLogtrans, &QAction::triggered, this, &MenuBar::LogTrans);
	connect(pGamma, &QAction::triggered, this, &MenuBar::Gamma);
	connect(pHisteq, &QAction::triggered, this, &MenuBar::Histeq);

	QMenu *pMenuNolinear = new QMenu(QString::fromLocal8Bit("非线性变换"));
	pMenuNolinear->addAction(pLogtrans);
	pMenuNolinear->addAction(pGamma);
	pMenuNolinear->addAction(pHisteq);

	QAction *pBin = new QAction(QIcon(), QString::fromLocal8Bit("二值图"), this);
	QAction *pGray = new QAction(QIcon(), QString::fromLocal8Bit("灰度图"), this);
	QAction *pReverse = new QAction(QIcon(), QString::fromLocal8Bit("反转变换"), this);
	QAction *pLinear = new QAction(QIcon(), QString::fromLocal8Bit("线性变换"), this);
	connect(pBin, &QAction::triggered, this, &MenuBar::Bin);
	connect(pGray, &QAction::triggered, this, &MenuBar::Gray);
	connect(pReverse, &QAction::triggered, this, &MenuBar::Reverse);
	connect(pLinear, &QAction::triggered, this, &MenuBar::Linear);

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

	QMenu *pSet = new QMenu(QString::fromLocal8Bit("主题"), this);
//	connect(pSet, &QAction::triggered, this, &MenuBar::Settings);

	QAction *pBlack = new QAction(QIcon(), QString::fromLocal8Bit("黑色"), this);
	QAction *pWhite = new QAction(QIcon(), QString::fromLocal8Bit("白色"), this);
	QAction *pGreen = new QAction(QIcon(), QString::fromLocal8Bit("健康绿"), this);
	connect(pBlack, &QAction::triggered, this, &MenuBar::Skin_Black);
	connect(pWhite, &QAction::triggered, this, &MenuBar::Skin_White);
	connect(pGreen, &QAction::triggered, this, &MenuBar::Skin_Green);

	pSet->addAction(pBlack);
	pSet->addAction(pWhite);
	pSet->addAction(pGreen);

	pHelp->addAction(pAbout);
	pHelp->addMenu(pSet);
}

void MenuBar::NewFile()
{
	qDebug() << "NewFile...";
	QImage img = QImage(500, 500, QImage::Format_RGB32);
	img.fill(qRgb(255, 255, 255));
	pParent_->RenderImage(img, true);
	pParent_->Report(QString::fromLocal8Bit("new img: 500*500"));
}

void MenuBar::OpenFile()
{
	QString strFile = QFileDialog::getOpenFileName(this, QString::fromLocal8Bit("select img "),
		".", QString::fromLocal8Bit("Images (*.jpg *.png *.bmp)"));
	if (!strFile.isEmpty())
	{
		QImage img;
		if (!img.load(strFile))
		{
			QMessageBox::information(this, QString::fromLocal8Bit("error"),
				QString::fromLocal8Bit("open img error!"));
		}
		pParent_->RenderImage(img, true);
		pParent_->Report("open img: " + strFile);
	}
}

void MenuBar::SaveFile()
{
	QString strPath = pParent_->GetImgWindowHandle()->GetPath();
	if (strPath.isEmpty())
	{
		SaveAs();
	}
	else {
		QImage img = pParent_->GetImgWindowHandle()->GetImg();
		img.save(strPath);
		pParent_->Report("save img" + strPath);
	}
}

void MenuBar::SaveAs()
{
	QString strFile = QFileDialog::getSaveFileName(this, "save file", ".",
		"Images (*.jpg *.png *.bmp)");
	if (!strFile.isEmpty())
	{
		QImage img = pParent_->GetImgWindowHandle()->GetImg();
		img.save(strFile);
		pParent_->Report("save img" + strFile);
	}
}

void MenuBar::Undo()
{
	qDebug() << "Undo...";
	HistoryMgr& historymgr = pParent_->GetHistoryMgr();
	if (historymgr.canUndo())
	{
		QImage curimg = historymgr.undo();
		pParent_->RenderImage(curimg, false);
	}
	UpdataUIState();
}

void MenuBar::Redo()
{
	qDebug() << "Redo...";

	HistoryMgr& historyMgr = pParent_->GetHistoryMgr();
	if (historyMgr.canRedo())
	{
		QImage curimg = historyMgr.redo();
		pParent_->RenderImage(curimg, false);
	}
	UpdataUIState();
}

void MenuBar::FullScreen()
{
	qDebug() << "FullScreen...";
	pParent_->showFullScreen();
}

void MenuBar::ExitFullScreen()
{
	qDebug() << "ExitFullScreen...";
	pParent_->showMaximized();
}

void MenuBar::Find()
{
	qDebug() << "Find...";
}

void MenuBar::ToolBox() 
{ 
	qDebug() << "ToolBox..."; 
	if (pParent_->GetToolWin()->isHidden())
	{
		pParent_->GetToolWin()->show();
		actionsMap_[QString::fromLocal8Bit("工具箱")]->setChecked(true);
	}
	else {
		pParent_->GetToolWin()->hide();
		actionsMap_[QString::fromLocal8Bit("工具箱")]->setChecked(false);
	}


}
void MenuBar::ImageWindow() { qDebug() << "ImageWindow..."; }
void MenuBar::OutputWindow() { qDebug() << "OutputWindow..."; }
void MenuBar::PropertyWindow() { qDebug() << "PropertyWindow..."; }
void MenuBar::FileToolBar() { qDebug() << "FileToolBar..."; }
void MenuBar::DrawToolBar() { qDebug() << "DrawToolBar..."; }


void MenuBar::AutoSize()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QScrollArea *pArea = pParent_->GetScrollArea();
	QDockWidget *pImgView = pParent_->GetImgView();
	QImage img = pImgWin->GetImg();
	QImage newimg;

	qDebug() << "img: " << img.width() << " * " << img.height();
	qDebug() << "imgwin: " << pImgWin->width() << " * " << pImgWin->height();
	newimg = img.scaled(pImgView->width(), pImgView->height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
	pParent_->RenderImage(newimg);
}

void MenuBar::Large()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().enlarge(srcimg, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::Small()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().enlarge(srcimg, -1);
	pParent_->RenderImage(newimg);
}

void MenuBar::Rotate()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().rodateByCenter(srcimg, 45);
	pParent_->RenderImage(newimg);
}

void MenuBar::RRotate()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().rodateByCenter(srcimg, -45);
	pParent_->RenderImage(newimg);
}

void MenuBar::CRotate()
{

}

void MenuBar::HFlip()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().flip(srcimg, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::VFlip()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().flip(srcimg, 0);
	pParent_->RenderImage(newimg);
}

void MenuBar::Bin()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().bin(srcimg, 127);
	pParent_->RenderImage(newimg);
}

void MenuBar::Gray()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().gray(srcimg);
	pParent_->RenderImage(newimg);
}

void MenuBar::Reverse()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Reverse(srcimg);
	pParent_->RenderImage(newimg);
}

void MenuBar::LogTrans()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().LogTrans(srcimg, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::Gamma()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Gamma(srcimg, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::Histeq()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Histeq(srcimg);
	pParent_->RenderImage(newimg);
}

void MenuBar::Linear()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Linear(srcimg, 50, 50);
	pParent_->RenderImage(newimg);
}


void MenuBar::CircleDetect()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().CircleDetect(srcimg, 1, 30);
	pParent_->RenderImage(newimg);
}

void MenuBar::LineDetect()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().LineDetect(srcimg );
	pParent_->RenderImage(newimg);
}

void MenuBar::Normalize()
{
	qDebug() << "Normalize...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Normalize(srcimg, 5);
	pParent_->RenderImage(newimg);
}

void MenuBar::Gaussian()
{
	qDebug() << "Gaussian...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Gaussian(srcimg, 5);
	pParent_->RenderImage(newimg);
}

void MenuBar::Median()
{
	qDebug() << "Median...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Median(srcimg, 5);
	pParent_->RenderImage(newimg);
}

void MenuBar::Sobel()
{
	qDebug() << "Sobel...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Sobel(srcimg, 5);
	pParent_->RenderImage(newimg);
}

void MenuBar::Laplacian()
{
	qDebug() << "Laplacian...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Laplacian(srcimg, 5);
	pParent_->RenderImage(newimg);
}

void MenuBar::Canny()
{
	qDebug() << "Canny...";
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Canny(srcimg, 5, 20, 80);
	pParent_->RenderImage(newimg);
}



void MenuBar::RGB_R()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().splitBGR(srcimg, 2);
	pParent_->RenderImage(newimg);
}

void MenuBar::RGB_B()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().splitBGR(srcimg, 0);
	pParent_->RenderImage(newimg);
}

void MenuBar::RGB_G()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().splitBGR(srcimg, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::HSV_H()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HSV", 0);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::HSV_S()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HSV", 1);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::HSV_V()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HSV", 2);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::YUV_Y()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "YUV", 0);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::YUV_U()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "YUV", 1);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::YUV_V()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "YUV", 2);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::HLS_H()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HLS", 0);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::HLS_L()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HLS", 1);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::HLS_S()
{
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	if (srcimg.depth() != 8)
	{
		QImage Img = pParent_->GetImgProCess().splitColor(srcimg, "HLS", 2);
		pParent_->RenderImage(Img);
	}
	else
	{
		QMessageBox message(QMessageBox::Information, tr("提示"), tr("该图像为灰度图像。"));
		message.exec();
	}
}

void MenuBar::Erode() 
{ 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Erode(srcimg, 0, 5, 1);
	pParent_->RenderImage(newimg);
}

void MenuBar::Dilate() 
{ 
	qDebug() << "Dilate..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().Dilate(srcimg, 0, 5, 1);
	pParent_->RenderImage(newimg);
}
void MenuBar::OpenOperation() 
{
	qDebug() << "OpenOperation..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().OpenOperation(srcimg, 0, 5, 1);
	pParent_->RenderImage(newimg);
}
void MenuBar::CloseOperation() 
{ 
	qDebug() << "CloseOperation..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().CloseOperation(srcimg, 0, 5, 1);
	pParent_->RenderImage(newimg);
}
void MenuBar::TopHat()
{
	qDebug() << "TopHat..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().TopHat(srcimg, 0, 5);
	pParent_->RenderImage(newimg);
}
void MenuBar::BlackHat() 
{
	qDebug() << "BlackHat..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().BlackHat(srcimg, 0, 5);
	pParent_->RenderImage(newimg);
}
void MenuBar::MorphologyGradient()
{ 
	qDebug() << "MorphologyGradient..."; 
	ImgWindow *pImgWin = pParent_->GetImgWindowHandle();
	QImage srcimg = pImgWin->GetImg();
	QImage newimg = pParent_->GetImgProCess().MorphologyGradient(srcimg, 0, 5);
	pParent_->RenderImage(newimg);

}

void MenuBar::About() 
{
	qDebug() << "About...";

	QMessageBox aboutBox(pParent_);
	aboutBox.setWindowTitle("img processtool v1.0");
	aboutBox.setText("Hello, this is a tool which to process the img. ");
	aboutBox.resize(600, 300);
	aboutBox.exec();
}
void MenuBar::Settings()
{ 
	qDebug() << "Settings...";



}

void MenuBar::Skin_Black()
{
	SkinManager::instance()->setSkin(SkinManager::Black);
}

void MenuBar::Skin_White()
{
	SkinManager::instance()->setSkin(SkinManager::White);
}

void MenuBar::Skin_Green()
{
	SkinManager::instance()->setSkin(SkinManager::Green);
}

void MenuBar::UpdataUIState()
{
	actionsMap_[QString::fromLocal8Bit("撤销")]->setEnabled(pParent_->GetHistoryMgr().canUndo());
	actionsMap_[QString::fromLocal8Bit("重做")]->setEnabled(pParent_->GetHistoryMgr().canRedo());
}
