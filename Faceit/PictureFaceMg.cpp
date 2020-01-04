#include "PictureFaceMg.h"
#include <QFileDialog>
#include <QTimer>
#include <QMovie>
#include <QPainter>
#include <QDebug>
#include <QMessageBox>
#include <QDebug>
#include <QDateTime>
#include <QMouseEvent>
#include <QtMultimedia/QMediaPlayer> 
#include <math.h>

PictureFaceMg::PictureFaceMg(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	//	加载用于人脸检测的级联分类器
	faceDetector.load("haarcascade_frontalface_alt2.xml");

	//	加载人脸形状探测器
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//	加载负责人脸识别的DNN
	dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	//	获得窗口长宽
	label_width = ui.label_input1->width();
	label_heigth = ui.label_input1->height();
}

PictureFaceMg::~PictureFaceMg()
{
}

//	检测出两张图片上的人脸关键点
void PictureFaceMg::faceLandmarkDetection(dlib::matrix<dlib::rgb_pixel>& img, 
		std::vector<cv::Point2f>& points, std::vector<cv::Rect> face) {

	//	获取脸部位置
	std::vector<dlib::rectangle> dlib_faces;	
	dlib::rectangle rect(face[0].x, face[0].y, face[0].x + face[0].width, face[0].y + face[0].height);
	dlib_faces.push_back(rect);
	//	将检测出的脸转为full_object_detection对象
	dlib::full_object_detection shape = sp(img, dlib_faces[0]);	

	//	获取脸部关键点坐标
	for (int i = 0; i < shape.num_parts(); ++i) {
		float x = shape.part(i).x();
		float y = shape.part(i).y();
		points.push_back(cv::Point2f(x, y));
	}
}

//	Delaunay 三角剖分
void PictureFaceMg::delaunayTriangulation(const std::vector<cv::Point2f>& hull,
		std::vector<correspondens>& delaunayTri, cv::Rect rect) {

	cv::Subdiv2D subdiv(rect);	//	细分三角剖分的数据单元
	for (int it = 0; it < hull.size(); it++)
		subdiv.insert(hull[it]);
	//	获得三角形列表
	std::vector<cv::Vec6f> triangleList;	
	subdiv.getTriangleList(triangleList);

	for (size_t i = 0; i < triangleList.size(); ++i)
	{
		std::vector<cv::Point2f> pt;	//	三角形顶点坐标
		correspondens ind;
		cv::Vec6f t = triangleList[i];
		pt.push_back(cv::Point2f(t[0], t[1]));
		pt.push_back(cv::Point2f(t[2], t[3]));
		pt.push_back(cv::Point2f(t[4], t[5]));

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			int count = 0;
			for (int j = 0; j < 3; ++j)
				for (size_t k = 0; k < hull.size(); k++)
					if (abs(pt[j].x - hull[k].x) < 1.0   &&  abs(pt[j].y - hull[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			if (count == 3) {
				delaunayTri.push_back(ind);	//	输出获得三角剖分的三角形
			}
		}
	}
}

//	将刚找到的仿射变换应用于src图像
void PictureFaceMg::applyAffineTransform(cv::Mat &warpImage, cv::Mat &src,
	std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri) {
	// 得到仿射变化
	cv::Mat warpMat = getAffineTransform(srcTri, dstTri);

	// 将刚找到的仿射变换应用于src图像
	cv::warpAffine(src, warpImage, warpMat, warpImage.size(),
		cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}

//	仿射变化
void PictureFaceMg::warpTriangle(cv::Mat &img1, cv::Mat &img2,
	std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2) {
	//	计算轮廓的最小外接矩形
	cv::Rect r1 = cv::boundingRect(t1);
	cv::Rect r2 = cv::boundingRect(t2);

	//	计算各个矩形的左上角偏移点
	std::vector<cv::Point2f> t1Rect, t2Rect;
	std::vector<cv::Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{
		t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	//	填充三角形得到掩码
	cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
	cv::fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

	//	将刚找到的仿射变换应用于src图像
	cv::Mat img1Rect;
	img1(r1).copyTo(img1Rect);
	cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	//	重新获得感兴趣区域
	cv::multiply(img2Rect, mask, img2Rect);
	cv::multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;
}

//	输入上图
void PictureFaceMg::inputImage1() {
	//	清空数据
	face1.clear();
	points1.clear();

	//	打开并设置图片
	QString path = QFileDialog::getOpenFileName(this,
		QStringLiteral("OpenImage1", "../", "Image(*.png *jpg *bmp)"));
	if (path.isEmpty()) {
		return;
	}
	QPixmap pixmap = QPixmap(path).scaled(label_width, label_heigth, 
		Qt::KeepAspectRatio, Qt::SmoothTransformation);

	ui.label_input1->setAlignment(Qt::AlignCenter);
	ui.label_input1->setPixmap(pixmap);

	image1 = cv::imread(path.toLocal8Bit().data());
	//	调整图片大小
	cv::resize(image1, image1, faceSize,0,0, cv::INTER_AREA);	
	//	将图片转为matrix的图片
	dlib::assign_image(img1, dlib::cv_image<dlib::rgb_pixel>(image1));

	faceDetector.detectMultiScale(image1, face1);	//	检测人脸区域

	if (face1.size() == 0 ) {
		QMessageBox::information(this, QStringLiteral("Warring！"), 
			QStringLiteral("上图未能识别出人脸，请重新选择！"));
		return;
	}

	//	检测出两张图片上的人脸关键点
	faceLandmarkDetection(img1, points1, face1);
}

//	输入下图
void PictureFaceMg::inputImage2() {
	//	清空数据
	face2.clear();
	points2.clear();

	//	打开并设置图片
	QString path = QFileDialog::getOpenFileName(this,
		QStringLiteral("OpenImage2", "../", "Image(*.png *jpg *bmp)"));
	if (path.isEmpty()) {
		return;
	}

	QPixmap pixmap = QPixmap(path);
	QPixmap fitpixmap = pixmap.scaled(label_width, label_heigth, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_input2->setAlignment(Qt::AlignCenter);
	ui.label_input2->setPixmap(fitpixmap);

	image2 = cv::imread(path.toLocal8Bit().data());

	image2_width = image2.cols;
	image2_heigth = image2.rows;

	cv::resize(image2, image2, faceSize, 0, 0, cv::INTER_AREA);

	dlib::assign_image(img2, dlib::cv_image<dlib::rgb_pixel>(image2));

	faceDetector.detectMultiScale(image2, face2);

	if (face2.size() == 0) {
		QMessageBox::information(this, QStringLiteral("Warring！"), QStringLiteral("下图未能识别出人脸，请重新选择！"));
		return;
	}

	//	检测出两张图片上的人脸关键点
	faceLandmarkDetection(img2, points2, face2);

}

//	人脸交换
void PictureFaceMg::faceSwap() {

	//	判断上下图是否能识别出人脸
	if (face1.size() < 0 || face2.size() < 0) {
		QMessageBox::information(this, QStringLiteral("Warring！"), 
			QStringLiteral("上图或下图未能识别出人脸，请重新选择！"));
		return;
	}

	//	进行图片的转换
	cv::Mat image2_swap = image2.clone();
	image1.convertTo(image1, CV_32F);
	image2_swap.convertTo(image2_swap, CV_32F);

	std::vector<cv::Point2f> hull1,hull2;	//	定义组成凸包的关键点
	std::vector<int> hullIndex;

	//	寻找图像的凸包
	cv::convexHull(points2, hullIndex, false, false);

	//	保存组成凸包的关键点
	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);
		hull2.push_back(points2[hullIndex[i]]);
	}

	std::vector<correspondens> delaunayTri;	//	三角剖分结果
	cv::Rect rect(0, 0, image2_swap.cols, image2_swap.rows);
	//	进行Delaunay三角剖分
	delaunayTriangulation(hull2, delaunayTri, rect);

	for (size_t i = 0; i < delaunayTri.size(); ++i)
	{
		std::vector<cv::Point2f> t1, t2;	//存放三角形的顶点
		correspondens corpd = delaunayTri[i];
		for (size_t j = 0; j < 3; ++j)
		{
			t1.push_back(hull1[corpd.index[j]]);
			t2.push_back(hull2[corpd.index[j]]);
		}
		//	进行仿射变换	
		warpTriangle(image1, image2_swap, t1, t2);	
	}

	std::vector<cv::Point> hull8U;	//	凸包坐标
	for (int i = 0; i < hull2.size(); ++i)
	{
		cv::Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}
	//	计算掩码
	cv::Mat mask = cv::Mat::zeros(image2.rows, image2.cols, image2.depth());
	cv::fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

	//	获得图二面部中心点坐标
	cv::Rect r = cv::boundingRect(hull2);
	cv::Point center = (r.tl() + r.br()) / 2;

	image2_swap.convertTo(image2_swap, CV_8UC3);
	//	使用泊松融合处理两张图边缘
	cv::seamlessClone(image2_swap, image2, mask, center, output, cv::NORMAL_CLONE);

	//	将结果图调整为图二原大小
	cv::resize(output, output, cv::Size(image2_width, image2_heigth), 0, 0, cv::INTER_CUBIC);
	
	//	使用filter2D改变内核达到轻微锐化效果
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	cv::filter2D(output, output, output.depth(), kernel);

	cv::imshow("Face Swapped", output);	//	输出结果图
}

//	人脸相似度对比
void PictureFaceMg::similarityDetection() {
	std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
	faces.clear();

	if (img1.size() == 0 || img2.size() == 0) {
		return;
	}

	//	加入获得的人脸
	faces.push_back(img1);
	faces.push_back(img2);

	//	使用DNN分析人脸数据
	std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);

	//	计算差距值
	float f = length(face_descriptors[0] - face_descriptors[1]);
	//	1-差距值则为相似度，相似度>=0.6则为同一个人
	float similarity = 1.0f - f;
	ui.label_similarity->setText(QString::number(similarity,'f',2));
}

//	保存图像
void PictureFaceMg::save() {
	QString path = QFileDialog::getSaveFileName(this,
		QStringLiteral("Face Swapped", "../", "Image(*.png *jpg *bmp)"));

	if (path.isEmpty()) {
		return;
	}
	else {
		if (cv::imwrite(path.toLocal8Bit().data(),output)) {
			QMessageBox::information(this, QStringLiteral("Tip！"), QStringLiteral("保存成功！"));
		}
	}
}
