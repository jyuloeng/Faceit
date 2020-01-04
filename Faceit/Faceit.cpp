#include "Faceit.h"

Faceit::Faceit(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

void Faceit::toPictureFaceMg() {
	view_PictureFaceMg = new PictureFaceMg();
	view_PictureFaceMg->show();
	this->close();
}

void Faceit::toVideoFaceMg() {
	view_VideoFaceMg = new VideoFaceMg();
	view_VideoFaceMg->show();
	this->close();
}

void Faceit::toVideoToChar() {
	view_VideoToChar = new VideoToChar();
	view_VideoToChar->show();
	this->close();
}

void Faceit::toPictureMg() {
	view_PictureMg = new PictureMg();
	view_PictureMg->show();
	this->close();
}

void VideoToChar::back() {
	Faceit *view_main = new Faceit();
	view_main->show();
	this->close();
}

void PictureFaceMg::back() {
	Faceit *view_main = new Faceit();
	view_main->show();
	this->close();
}

void VideoFaceMg::back() {
	Faceit *view_main = new Faceit();
	view_main->show();
	this->close();
}

void PictureMg::back() {
	Faceit *view_main = new Faceit();
	view_main->show();
	this->close();
}