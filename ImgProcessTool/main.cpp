#include "ImgProcessTool.h"
#include <QtWidgets/QApplication>
#include <QTextCodec>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//	QTextCodec *codec = QTextCodec::codecForName("UTF-8");
//	QTextCodec::setCodecForCStrings(codec);
    ImgProcessTool w;
    w.show();
    return a.exec();
}
