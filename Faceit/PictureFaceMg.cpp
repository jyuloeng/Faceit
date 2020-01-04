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

	//	���������������ļ���������
	faceDetector.load("haarcascade_frontalface_alt2.xml");

	//	����������״̽����
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//	���ظ�������ʶ���DNN
	dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	//	��ô��ڳ���
	label_width = ui.label_input1->width();
	label_heigth = ui.label_input1->height();
}

PictureFaceMg::~PictureFaceMg()
{
}

//	��������ͼƬ�ϵ������ؼ���
void PictureFaceMg::faceLandmarkDetection(dlib::matrix<dlib::rgb_pixel>& img, 
		std::vector<cv::Point2f>& points, std::vector<cv::Rect> face) {

	//	��ȡ����λ��
	std::vector<dlib::rectangle> dlib_faces;	
	dlib::rectangle rect(face[0].x, face[0].y, face[0].x + face[0].width, face[0].y + face[0].height);
	dlib_faces.push_back(rect);
	//	����������תΪfull_object_detection����
	dlib::full_object_detection shape = sp(img, dlib_faces[0]);	

	//	��ȡ�����ؼ�������
	for (int i = 0; i < shape.num_parts(); ++i) {
		float x = shape.part(i).x();
		float y = shape.part(i).y();
		points.push_back(cv::Point2f(x, y));
	}
}

//	Delaunay �����ʷ�
void PictureFaceMg::delaunayTriangulation(const std::vector<cv::Point2f>& hull,
		std::vector<correspondens>& delaunayTri, cv::Rect rect) {

	cv::Subdiv2D subdiv(rect);	//	ϸ�������ʷֵ����ݵ�Ԫ
	for (int it = 0; it < hull.size(); it++)
		subdiv.insert(hull[it]);
	//	����������б�
	std::vector<cv::Vec6f> triangleList;	
	subdiv.getTriangleList(triangleList);

	for (size_t i = 0; i < triangleList.size(); ++i)
	{
		std::vector<cv::Point2f> pt;	//	�����ζ�������
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
				delaunayTri.push_back(ind);	//	�����������ʷֵ�������
			}
		}
	}
}

//	�����ҵ��ķ���任Ӧ����srcͼ��
void PictureFaceMg::applyAffineTransform(cv::Mat &warpImage, cv::Mat &src,
	std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri) {
	// �õ�����仯
	cv::Mat warpMat = getAffineTransform(srcTri, dstTri);

	// �����ҵ��ķ���任Ӧ����srcͼ��
	cv::warpAffine(src, warpImage, warpMat, warpImage.size(),
		cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}

//	����仯
void PictureFaceMg::warpTriangle(cv::Mat &img1, cv::Mat &img2,
	std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2) {
	//	������������С��Ӿ���
	cv::Rect r1 = cv::boundingRect(t1);
	cv::Rect r2 = cv::boundingRect(t2);

	//	����������ε����Ͻ�ƫ�Ƶ�
	std::vector<cv::Point2f> t1Rect, t2Rect;
	std::vector<cv::Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{
		t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	//	��������εõ�����
	cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
	cv::fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

	//	�����ҵ��ķ���任Ӧ����srcͼ��
	cv::Mat img1Rect;
	img1(r1).copyTo(img1Rect);
	cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	//	���»�ø���Ȥ����
	cv::multiply(img2Rect, mask, img2Rect);
	cv::multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;
}

//	������ͼ
void PictureFaceMg::inputImage1() {
	//	�������
	face1.clear();
	points1.clear();

	//	�򿪲�����ͼƬ
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
	//	����ͼƬ��С
	cv::resize(image1, image1, faceSize,0,0, cv::INTER_AREA);	
	//	��ͼƬתΪmatrix��ͼƬ
	dlib::assign_image(img1, dlib::cv_image<dlib::rgb_pixel>(image1));

	faceDetector.detectMultiScale(image1, face1);	//	�����������

	if (face1.size() == 0 ) {
		QMessageBox::information(this, QStringLiteral("Warring��"), 
			QStringLiteral("��ͼδ��ʶ���������������ѡ��"));
		return;
	}

	//	��������ͼƬ�ϵ������ؼ���
	faceLandmarkDetection(img1, points1, face1);
}

//	������ͼ
void PictureFaceMg::inputImage2() {
	//	�������
	face2.clear();
	points2.clear();

	//	�򿪲�����ͼƬ
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
		QMessageBox::information(this, QStringLiteral("Warring��"), QStringLiteral("��ͼδ��ʶ���������������ѡ��"));
		return;
	}

	//	��������ͼƬ�ϵ������ؼ���
	faceLandmarkDetection(img2, points2, face2);

}

//	��������
void PictureFaceMg::faceSwap() {

	//	�ж�����ͼ�Ƿ���ʶ�������
	if (face1.size() < 0 || face2.size() < 0) {
		QMessageBox::information(this, QStringLiteral("Warring��"), 
			QStringLiteral("��ͼ����ͼδ��ʶ���������������ѡ��"));
		return;
	}

	//	����ͼƬ��ת��
	cv::Mat image2_swap = image2.clone();
	image1.convertTo(image1, CV_32F);
	image2_swap.convertTo(image2_swap, CV_32F);

	std::vector<cv::Point2f> hull1,hull2;	//	�������͹���Ĺؼ���
	std::vector<int> hullIndex;

	//	Ѱ��ͼ���͹��
	cv::convexHull(points2, hullIndex, false, false);

	//	�������͹���Ĺؼ���
	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);
		hull2.push_back(points2[hullIndex[i]]);
	}

	std::vector<correspondens> delaunayTri;	//	�����ʷֽ��
	cv::Rect rect(0, 0, image2_swap.cols, image2_swap.rows);
	//	����Delaunay�����ʷ�
	delaunayTriangulation(hull2, delaunayTri, rect);

	for (size_t i = 0; i < delaunayTri.size(); ++i)
	{
		std::vector<cv::Point2f> t1, t2;	//��������εĶ���
		correspondens corpd = delaunayTri[i];
		for (size_t j = 0; j < 3; ++j)
		{
			t1.push_back(hull1[corpd.index[j]]);
			t2.push_back(hull2[corpd.index[j]]);
		}
		//	���з���任	
		warpTriangle(image1, image2_swap, t1, t2);	
	}

	std::vector<cv::Point> hull8U;	//	͹������
	for (int i = 0; i < hull2.size(); ++i)
	{
		cv::Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}
	//	��������
	cv::Mat mask = cv::Mat::zeros(image2.rows, image2.cols, image2.depth());
	cv::fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

	//	���ͼ���沿���ĵ�����
	cv::Rect r = cv::boundingRect(hull2);
	cv::Point center = (r.tl() + r.br()) / 2;

	image2_swap.convertTo(image2_swap, CV_8UC3);
	//	ʹ�ò����ںϴ�������ͼ��Ե
	cv::seamlessClone(image2_swap, image2, mask, center, output, cv::NORMAL_CLONE);

	//	�����ͼ����Ϊͼ��ԭ��С
	cv::resize(output, output, cv::Size(image2_width, image2_heigth), 0, 0, cv::INTER_CUBIC);
	
	//	ʹ��filter2D�ı��ں˴ﵽ��΢��Ч��
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	cv::filter2D(output, output, output.depth(), kernel);

	cv::imshow("Face Swapped", output);	//	������ͼ
}

//	�������ƶȶԱ�
void PictureFaceMg::similarityDetection() {
	std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
	faces.clear();

	if (img1.size() == 0 || img2.size() == 0) {
		return;
	}

	//	�����õ�����
	faces.push_back(img1);
	faces.push_back(img2);

	//	ʹ��DNN������������
	std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);

	//	������ֵ
	float f = length(face_descriptors[0] - face_descriptors[1]);
	//	1-���ֵ��Ϊ���ƶȣ����ƶ�>=0.6��Ϊͬһ����
	float similarity = 1.0f - f;
	ui.label_similarity->setText(QString::number(similarity,'f',2));
}

//	����ͼ��
void PictureFaceMg::save() {
	QString path = QFileDialog::getSaveFileName(this,
		QStringLiteral("Face Swapped", "../", "Image(*.png *jpg *bmp)"));

	if (path.isEmpty()) {
		return;
	}
	else {
		if (cv::imwrite(path.toLocal8Bit().data(),output)) {
			QMessageBox::information(this, QStringLiteral("Tip��"), QStringLiteral("����ɹ���"));
		}
	}
}
