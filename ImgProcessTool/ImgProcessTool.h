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

class ImgProcessTool : public QMainWindow
{
    Q_OBJECT
public:
	struct PropertyControl {
		QLineEdit *pLineImg;
		QLineEdit *pLineLen;
		QLineEdit *pLineWid;
		QLineEdit *pLineDepth;
		QLineEdit *pLineGray;
	};

    ImgProcessTool(QWidget *parent = Q_NULLPTR);

	void InitForm();

	void InitLayOut();
	void InitImgView();
	void InitUiView();
	void InitPainterView();
	void InitPropertyView();
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


};
