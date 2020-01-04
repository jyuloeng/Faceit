#pragma once

#include <QWidget>
#include "ui_PictureFaceMg.h"
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

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;

struct correspondens {
	std::vector<int> index;
};


class PictureFaceMg : public QWidget
{
	Q_OBJECT

public:
	PictureFaceMg(QWidget *parent = Q_NULLPTR);
	~PictureFaceMg();

public:
	Ui::PictureFaceMg ui;

	cv::CascadeClassifier faceDetector;	//	用于人脸检测的级联分类器
	dlib::shape_predictor sp;	//	人脸形状探测器
	anet_type net;				//	负责人脸识别的DNN

	cv::Mat image1;		//	输入上图（作为面部提取）
	cv::Mat image2;		//	输入下图（作为交换背景）
	cv::Mat output;		//	脸部交换结果

	//	获得窗口长宽
	int label_width;
	int label_heigth;

	//	获得下图的长宽
	int image2_width;
	int image2_heigth;

	//	dlib库matrix类型的上、下图
	dlib::matrix<dlib::rgb_pixel> img1;
	dlib::matrix<dlib::rgb_pixel> img2;

	//	脸部区域
	std::vector<cv::Rect> face1;
	std::vector<cv::Rect> face2;

	std::vector<cv::Point2f> points1, points2;	//	脸部关键点坐标

	cv::Size faceSize = cv::Size(150, 150);		//	处理脸部的大小

	//	检测出两张图片上的人脸关键点
	void faceLandmarkDetection(dlib::matrix<dlib::rgb_pixel>& img,
		std::vector<cv::Point2f>& points, std::vector<cv::Rect> face);

	//	Delaunay 三角剖份
	void delaunayTriangulation(const std::vector<cv::Point2f>& hull,
		std::vector<correspondens>& delaunayTri, cv::Rect rect);

	//	进行仿射变化
	void warpTriangle(cv::Mat &img1, cv::Mat &img2, 
		std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2);
	
	//	将刚找到的仿射变换应用于输入图像
	void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src,
		std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri);

private slots:
	void inputImage1();
	void inputImage2();
	void faceSwap();
	void similarityDetection();
	void save();
	void back();
};
