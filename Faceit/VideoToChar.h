#pragma once

#include <QWidget>
#include "ui_VideoToChar.h"
#include <fstream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include<math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <ctime>

class VideoToChar : public QWidget
{
	Q_OBJECT

public:
	VideoToChar(QWidget *parent = Q_NULLPTR);
	~VideoToChar();

private:
	Ui::VideoToChar ui;

	//	���ڳ���
	int window_width;
	int window_heigth;

	cv::Mat frame;	//	��ǰ֡
	cv::Mat temp;	//	��ʱ֡

	cv::VideoCapture capture;	//	��Ƶ����
	//	��ʱ��(ͨ����ʱ������������һ֡�Ӷ��õ����ŵ�Ч����1000/35~28֡����һ��28֡
	QTimer *frameTimer;			

	void mediaInLabel();	//	��label������Ƶ
	void getNextProcessFrame();		//	�����һ����֡

private slots:
	void back();
	void openVideo();
};