#pragma once

#include <QWidget>
#include "ui_VideoFaceMg.h"
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

class VideoFaceMg : public QWidget
{
	Q_OBJECT

public:
	VideoFaceMg(QWidget *parent = Q_NULLPTR);
	~VideoFaceMg();

private:
	Ui::VideoFaceMg ui;

	//	窗口长宽
	int window_width;	
	int window_heigth;

	cv::CascadeClassifier faceDetector;	//	用于人脸检测的级联分类器
	dlib::shape_predictor sp;	//	dlib人脸特征点检测器

	cv::Mat frame;	//	当前帧
	cv::Mat temp;	//	临时帧

	cv::Mat faceSrc;	//	待换图
	cv::Mat faceGray;	//	待换图的灰度图

	cv::Mat mixed;	//	混合帧

	cv::Mat roi;	//	脸部区域1
	cv::Mat roi2;	//	脸部区域2

	//	视频人脸中心坐标
	double face2_center_x;
	double face2_center_y;

	//	haar级联分类器探测人脸区域，获取一系列人脸所在区域
	std::vector<cv::Rect> face1;
	std::vector<cv::Rect> face2;

	cv::VideoCapture capture;	//	摄像头对象
	cv::VideoWriter writer;		//	写视频对象
	//	定时器(通过定时器获得来获得下一帧从而得到播放的效果）1000/35~28帧，既一秒28帧
	QTimer *frameTimer;			

	QImage screenImage;	//	屏幕截图

	bool isRecord = false;		//	是否正在录制
	bool isCameraOrVideoFlag;	//	true代表Camera,false代表Video

	void mediaToLabel();	//	在label播放视频
	void frameProcess(cv::Mat &output);		//	人脸识别帧处理函数
	void faceSwapProcess(cv::Mat &output);	//	人脸交换帧处理函数
	void faceLine(cv::Mat img, 
		std::vector<dlib::full_object_detection> fs);	//	描出轮廓

private slots:
	void openVideo();
	void closeVideo();
	void openCamera();
	void closeCamera();
	void printScreen();
	void recordVideo();
	void finishRecord();
	void faceRecognition();
	void faceSwapInput();
	void faceSwap();
	void back();

	void getNextFrame();	//	获取下一帧
	void getProcessFrame();	//	获取下一处理帧
	void getFaceSwapFrame();//	获取下一脸部交换帧
};
