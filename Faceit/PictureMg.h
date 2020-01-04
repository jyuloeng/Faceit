#pragma once

#include <QWidget>
#include "ui_PictureMg.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <ctime>

class PictureMg : public QWidget
{
	Q_OBJECT

public:
	PictureMg(QWidget *parent = Q_NULLPTR);
	~PictureMg();

private:
	Ui::PictureMg ui;

	cv::Mat image;		//	����ͼ
	cv::Mat gray;		//	����ͼ�Ҷ�ͼ
	cv::Mat imageClone;	//	����ͼ��¡ͼ
	cv::Mat result;		//	Ԥ��ͼ

	std::string nameWindow = "Ԥ��ͼ";
	QString path;		//	����ͼ·��

	int liangduValue = 0;
	int duibiduValue = 100;

	void loadImage();	//	����ͼƬ
	void showMessage();	//	��ʾͼ����Ϣ

private slots:
	void open();
	void save();
	void reset();
	void fanse();
	void fudiao();
	void manhua();
	void sumiao();
	void jingxiang();
	void fugu();
	void miaobian();
	void heibai();
	void suoxiao();
	void fangda();
	void shuzi();
	void bolang();
	void back();
};
