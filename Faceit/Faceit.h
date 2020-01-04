#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Faceit.h"
#include "PictureFaceMg.h"
#include "PictureMg.h"
#include "VideoFaceMg.h"
#include "VideoToChar.h"

class Faceit : public QMainWindow
{
	Q_OBJECT

public:
	Faceit(QWidget *parent = Q_NULLPTR);

private:
	Ui::FaceitClass ui;
	PictureFaceMg *view_PictureFaceMg;
	PictureMg *view_PictureMg;
	VideoFaceMg *view_VideoFaceMg;
	VideoToChar *view_VideoToChar;


private slots:
	void toPictureFaceMg();
	void toVideoFaceMg();
	void toVideoToChar();
	void toPictureMg();
};
