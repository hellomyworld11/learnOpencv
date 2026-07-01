#pragma once

#include <QMenuBar>
#include <QMap>
class ImgProcessTool;


class MenuBar : public QMenuBar
{
	Q_OBJECT

public:
	MenuBar(QWidget *parent);

	~MenuBar();

	void Init();
	
	void Menu_File();

	void Menu_Edit();

	void Menu_View();

	void Menu_GeoTransform();

	void Menu_GrayTransform();

	void Menu_ImageEnhance();

	void Menu_ColorModel();

	void Menu_Morphology();

	void Menu_Help();

	QAction* GetAction(QString actionname) { return actionsMap_[actionname]; }

	void UpdataUIState();
private slots:
	//文件子菜单
	void NewFile();
	
	void OpenFile();

	void SaveFile();

	void SaveAs();

	// 编辑
	void Undo();

	void Redo();

	void FullScreen();

	void ExitFullScreen();

	void Find();

	// 视图
	void ToolBox();

	void ImageWindow();

	void OutputWindow();

	void PropertyWindow();

	void FileToolBar();

	void DrawToolBar();

	// 几何变换
	void AutoSize();

	void Large();

	void Small();

	void Rotate();

	void RRotate();

	void CRotate();

	void HFlip();

	void VFlip();
	
	// 灰度变换
	void Bin();

	void Gray();

	void Reverse();

	void LogTrans();

	void Gamma();

	void Histeq();

	void Linear();

	// 图像增强
	void CircleDetect();

	void LineDetect();

	void Normalize();

	void Gaussian();

	void Median();

	void Sobel();

	void Laplacian();

	void Canny();

	// 颜色模型
	void RGB_R();

	void RGB_B();

	void RGB_G();

	void HSV_H();

	void HSV_S();

	void HSV_V();

	void YUV_Y();

	void YUV_U();

	void YUV_V();

	void HLS_H();

	void HLS_L();

	void HLS_S();

	// 形态学
	void Erode();

	void Dilate();

	void OpenOperation();

	void CloseOperation();

	void TopHat();

	void BlackHat();

	void MorphologyGradient();

	// 帮助
	void About();

	void Settings();

	void Skin_Black();

	void Skin_White();

	void Skin_Green();

private:
	
	ImgProcessTool *pParent_ = nullptr;

	QMap<QString, QAction*> actionsMap_;
};
