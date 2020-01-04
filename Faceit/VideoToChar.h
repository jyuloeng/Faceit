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

	//	窗口长宽
	int window_width;
	int window_heigth;

	cv::Mat frame;	//	当前帧
	cv::Mat temp;	//	临时帧

	cv::VideoCapture capture;	//	视频对象
	//	定时器(通过定时器获得来获得下一帧从而得到播放的效果）1000/35~28帧，既一秒28帧
	QTimer *frameTimer;			

	void mediaInLabel();	//	在label播放视频
	void getNextProcessFrame();		//	获得下一处理帧

private slots:
	void back();
	void openVideo();
};