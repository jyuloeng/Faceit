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

	//	���ڳ���
	int window_width;	
	int window_heigth;

	cv::CascadeClassifier faceDetector;	//	�����������ļ���������
	dlib::shape_predictor sp;	//	dlib��������������

	cv::Mat frame;	//	��ǰ֡
	cv::Mat temp;	//	��ʱ֡

	cv::Mat faceSrc;	//	����ͼ
	cv::Mat faceGray;	//	����ͼ�ĻҶ�ͼ

	cv::Mat mixed;	//	���֡

	cv::Mat roi;	//	��������1
	cv::Mat roi2;	//	��������2

	//	��Ƶ������������
	double face2_center_x;
	double face2_center_y;

	//	haar����������̽���������򣬻�ȡһϵ��������������
	std::vector<cv::Rect> face1;
	std::vector<cv::Rect> face2;

	cv::VideoCapture capture;	//	����ͷ����
	cv::VideoWriter writer;		//	д��Ƶ����
	//	��ʱ��(ͨ����ʱ������������һ֡�Ӷ��õ����ŵ�Ч����1000/35~28֡����һ��28֡
	QTimer *frameTimer;			

	QImage screenImage;	//	��Ļ��ͼ

	bool isRecord = false;		//	�Ƿ�����¼��
	bool isCameraOrVideoFlag;	//	true����Camera,false����Video

	void mediaToLabel();	//	��label������Ƶ
	void frameProcess(cv::Mat &output);		//	����ʶ��֡������
	void faceSwapProcess(cv::Mat &output);	//	��������֡������
	void faceLine(cv::Mat img, 
		std::vector<dlib::full_object_detection> fs);	//	�������

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

	void getNextFrame();	//	��ȡ��һ֡
	void getProcessFrame();	//	��ȡ��һ����֡
	void getFaceSwapFrame();//	��ȡ��һ��������֡
};
