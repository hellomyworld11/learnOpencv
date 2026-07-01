#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_ImgProcessTool.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include "MenuBar.h"
#include <QDockWidget>
#include <QLineEdit>
#include <QVector>
#include <QTextEdit>
#include "ImgWindow.h"
#include "ImgProcess.h"
#include "QScrollArea"
#include "HistoryMgr.h"

class ImgProcessTool : public QMainWindow
{
    Q_OBJECT
public:
	struct PropertyControl {
		QLineEdit *pLineImg = nullptr;
		QLineEdit *pLineLen = nullptr;
		QLineEdit *pLineWid = nullptr;
		QLineEdit *pLineGray = nullptr;
		QLineEdit *pLineDepth = nullptr;
		bool Update(QImage& img);
	};

	enum class ToolBtnId{
		Pen=1,Line,Circle,Ellipse,Triangle,Rhombus,Rect,Square,Six,Pix
	};

    ImgProcessTool(QWidget *parent = Q_NULLPTR);

	void InitForm();

	void InitLayOut();

	void InitImgView();

	void InitUiView();

	void InitPainterView();

	void InitPropertyView();

	void RenderImage(QImage Img, bool bPushState = true);

	void Report(QString str);

	ImgWindow* GetImgWindowHandle() { return imgwin_; }

	QDockWidget *GetToolWin() { return pPainterView_; }

	QDockWidget* GetImgView() { return pImgView_; }

	HistoryMgr& GetHistoryMgr() { return imgHistory_; }

	ImgProcess& GetImgProCess() {
		return imgProcess_;
	}

	QScrollArea* GetScrollArea() {
		return ImgscrollArea;
	}

	void PushState();
private slots:
	void on_ToolButtonClicked(int id);
private:
    Ui::ImgProcessToolClass ui;

	MenuBar *pMenuBar_;

	QDockWidget *pImgView_;

	QDockWidget *pOutPutView_;

	QDockWidget *pPainterView_;

	QDockWidget *pPropertyView_;

	PropertyControl propertyInfo_;

	QTextEdit *pOutputEdit_;

	ImgWindow *imgwin_;

	ImgProcess imgProcess_;

	QScrollArea* ImgscrollArea;					// Í¼Ïñ´°¿Ú»¬¶¯Ìõ

	HistoryMgr imgHistory_;
};
