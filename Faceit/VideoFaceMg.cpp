#include "VideoFaceMg.h"
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

VideoFaceMg::VideoFaceMg(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	//	�����������ļ���������
	faceDetector.load("haarcascade_frontalface_alt2.xml");

	//	����dlib��������������
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//	������ڲ�����Ƶ��Label�Ŀ��
	window_heigth = ui.label_media->height();	
	window_width = ui.label_media->width();

	frameTimer = new QTimer(this);	//	��ʼ����ʱ��
}

VideoFaceMg::~VideoFaceMg()
{
}

void VideoFaceMg::openVideo() {
	isCameraOrVideoFlag = false;	//	false����Ϊ��Ƶ�ļ�

	QString srcPath = QFileDialog::getOpenFileName(this,
		"����Ƶ", "../", "Video(*.mp4 *.flv *.mkv *.avi)");
	if (srcPath.isEmpty()) {
		return;
	}
	//	��ʾ��label��
	ui.label_media->show();
	capture.open(srcPath.toLocal8Bit().data());

	frameTimer->destroyed();	//	����ʱ������

	//	�ж϶�ʱ������״̬
	if (frameTimer->isActive() == false) {
		//	������ʱ��
		frameTimer->start(35);
	}
	//	���Ӵ�����һ֡�ĺ������ź�
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getNextFrame);
}

void VideoFaceMg::closeVideo() {
	capture.release();	//	�ͷ���Դ
	frameTimer->stop();	//	ֹͣ��ʱ��

	ui.label_media->close();
}

void VideoFaceMg::openCamera() {
	isCameraOrVideoFlag = true;

	ui.label_media->show();
	capture.open(0);

	frameTimer->destroyed();

	//	�ж϶�ʱ������״̬
	if (frameTimer->isActive() == false) {
		//	������ʱ��
		frameTimer->start(35);
	}

	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getNextFrame);
}

void VideoFaceMg::closeCamera() {
	capture.release();
	frameTimer->stop();

	ui.label_media->close();
}

void VideoFaceMg::printScreen() {
	//	��Mat��tempתΪQt��QImage��
	screenImage = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format::Format_RGB888);

	ui.label_printScreen->setPixmap(QPixmap::fromImage(screenImage));
	ui.label_printScreen->setScaledContents(true);

	//	���浽����
	const QPixmap *screenShot = ui.label_printScreen->pixmap();
	QString path = QFileDialog::getSaveFileName(this, "save", "../", "Image(*.jpg)");
	if (path.isEmpty()) {
		return;
	}
	else {
		if (screenShot->save(path)) {
			QMessageBox::information(this, 
				QStringLiteral("Tip��"), QStringLiteral("��ͼ����ɹ���"));
		}
	}
}

void VideoFaceMg::recordVideo() {
	double rate = 6.0;	//	����¼��֡��
	cv::Size videoSize(frame.cols, frame.rows);	//	���ô�С

	//	����д��Ƶ�ļ�����
	writer.open("RecordVideo.mp4", 
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), rate, videoSize, true);
	isRecord = true;

	while (isRecord) {
		capture.read(frame);

		//	�������
		cv::flip(frame, frame, 1);
		writer.write(frame);	//	д��Ƶ�ļ�
		cv::imshow("¼����...", frame);
		cv::waitKey(35);
	}
}

void VideoFaceMg::finishRecord() {
	cv::destroyWindow("¼����...");	//	�رմ���
	isRecord = false;
	writer.release();	//	�ͷ���Դ
}

void VideoFaceMg::faceRecognition() {
	//	���ӻ�ȡ��һ����ʶ����֡���ź�
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getProcessFrame);
}

//	�����������
void VideoFaceMg::faceSwapInput() {
	//	�򿪲�����ͼƬ
	QString path = QFileDialog::getOpenFileName(this,
		QStringLiteral("OpenImage1", "../", "Image(*.png *jpg *bmp)"));
	if (path.isEmpty()) {
		return;
	}

	faceSrc = cv::imread(path.toLocal8Bit().data());

	//	��ͼ�����һϵ�д�������ȡ�Ϻõ�����
	cv::cvtColor(faceSrc, faceGray, CV_BGR2GRAY);	
	cv::equalizeHist(faceGray, faceGray);	//	ֱ��ͼ���������Աȶ�
	cv::medianBlur(faceGray, faceGray, 3);	//	��ֵ�˲���΢����

	QPixmap pixmap = QPixmap(path).scaled(ui.label_faceSwapInput->width(), ui.label_faceSwapInput->height(),
		Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_faceSwapInput->setAlignment(Qt::AlignCenter);
	ui.label_faceSwapInput->setPixmap(pixmap);
}

//	��������
void VideoFaceMg::faceSwap() {
	//	���ӻ�ȡ��һ������������֡���ź�
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getFaceSwapFrame);
}

//	��label��������Ƶ
void VideoFaceMg::mediaToLabel() {
	if (isCameraOrVideoFlag == true) {
		cv::flip(frame, frame, 1);
	}

	cv::cvtColor(frame, temp, CV_BGR2RGB);

	//	תΪQImage��ʽ
	QImage image = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format_RGB888);

	QPixmap pixmap = QPixmap::fromImage(image);
	QPixmap fitpixmap = pixmap.scaled(window_width, window_heigth, Qt::KeepAspectRatio, Qt::SmoothTransformation);

	ui.label_media->setAlignment(Qt::AlignCenter);
	ui.label_media->setPixmap(fitpixmap);
}

//	����ʶ��֡������
void VideoFaceMg::frameProcess(cv::Mat &output) {

	cv::Mat gray;	
	cv::cvtColor(output, gray, cv::COLOR_BGR2GRAY);	
	cv::equalizeHist(gray, gray);	//	ֱ��ͼ���⻯

	//	Mat����ת��Ϊdlib��array2d����
	dlib::array2d<dlib::bgr_pixel> dlib_img;
	dlib::assign_image(dlib_img, dlib::cv_image<uchar>(gray));

	std::vector<cv::Rect> cv_faces;			//	opencv�����͵���������
	std::vector<dlib::rectangle> dlib_faces;//	  dlib�����͵���������

	//	haar����������̽���������򣬻�ȡһϵ��������������
	faceDetector.detectMultiScale(gray, cv_faces);

	for (int i = 0; i < cv_faces.size(); i++)
	{
		cv::rectangle(output, cv_faces[i], CV_RGB(200, 0, 0));
		dlib::rectangle rect(cv_faces[i].x, cv_faces[i].y,
			cv_faces[i].x + cv_faces[i].width, cv_faces[i].y + cv_faces[i].height);
		dlib_faces.push_back(rect);
	}

	//	��ȡ����68�������㲿λ�ֲ�(�õ�ÿһ����������shapes��)
	std::vector<dlib::full_object_detection> shapes;
	for (int i = 0; i < dlib_faces.size(); i++)
	{
		dlib::full_object_detection shape = sp(dlib_img, dlib_faces[i]);
		shapes.push_back(shape);
	}

	//	��������
	faceLine(output, shapes);
}

//	��������֡������
void VideoFaceMg::faceSwapProcess(cv::Mat &output) {
	//	�ȼ��ͼ����������ΪҪ��ͼһ��ȡ��roi���ͼ���Ĵ�С	��ͼ��Ϊ��Ƶ��ǰ֡��
	faceDetector.detectMultiScale(output, face2);		//	���ͼ������
	if (face2.size() == 0) {}
	else {	//	��ͼ����������
		for (int i = 0; i < face2.size(); i++)
		{
			roi2 = output(face2[i]);	//	���ͼ����������
			//	���ͼ��������������
			face2_center_x = cvRound((face2[i].x + face2[i].width * 0.5));
			face2_center_y = cvRound((face2[i].y + face2[i].height * 0.5));
		}
		
		faceDetector.detectMultiScale(faceGray, face1);	//	���ͼһ����
		if (face1.size() == 0) {
			QMessageBox::information(this, QStringLiteral("Warring��"), 
				QStringLiteral("����ͼδ��ʶ���������������ѡ��"));
			return;
		}
		for (int i = 0; i < face1.size(); i++)
		{
			roi = faceSrc(face1[i]);	//	���ͼһ�������򲢵�����С
			cv::resize(roi, roi, cv::Size(roi2.cols*0.95, roi2.rows*0.95));
		}

		cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);	//	��������

		cv::Point face1_center(mask.cols / 2, mask.rows / 2);
		//	face1���沿��С�뾶
		int radius = std::min(mask.cols, mask.rows) / 2.3;

		cv::ellipse(mask, face1_center,		//	ʹ����Բ��������
			cv::Size(radius*0.9, radius*1.05), 0, -180, 180, cv::Scalar(255), -1);

		roi.copyTo(roi, mask);

		cv::Point face2_center(face2_center_x, face2_center_y);

		//	����Ӧλ�ý����ɰ��ں�
		seamlessClone(roi, output, mask, face2_center, output, cv::NORMAL_CLONE);
	}
}

//	��������
void VideoFaceMg::faceLine(cv::Mat img, std::vector<dlib::full_object_detection> faces)
{
	int i, j;
	for (j = 0; j < faces.size(); j++)
	{
		cv::Point startPoint, endPoint;	//	����������յ�

		for (i = 0; i < 67; i++)
		{
			switch (i)
			{
			case 16:	//�°͵����� 0 ~ 16
			case 21:	//���üë	17 ~ 21
			case 26:	//�ұ�üë	21 ~ 26
			case 30:	//����		27 ~ 30
			case 35:	//�ǿ�		31 ~ 35
			case 41:	//����		36 ~ 41
			case 47:	//����		42 ~ 47
			case 59:	//�촽��Ȧ	48 ~ 59 �촽��Ȧ	59 ~ 67
				i++;
				break;
			default:
				break;
			}

			startPoint.x = faces[j].part(i).x();
			startPoint.y = faces[j].part(i).y();
			endPoint.x = faces[j].part(i + 1).x();
			endPoint.y = faces[j].part(i + 1).y();
			//	line�������л���
			cv::line(img, startPoint, endPoint, cv::Scalar(0, 0, 255), 1);
		}
	}
}

//	��ȡ��һ֡
void VideoFaceMg::getNextFrame() {
	//	�ж��Ƿ�����һ֡
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	//	��label��������Ƶ
	mediaToLabel();
}

//	�����һ����ʶ����֡
void VideoFaceMg::getProcessFrame() {
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	frameProcess(frame);	//	����ʶ��֡������
	mediaToLabel();			//	��label��������Ƶ
}

//	�����һ������������֡
void VideoFaceMg::getFaceSwapFrame() {
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	faceSwapProcess(frame);	//	��������֡������
	mediaToLabel();			//	��label��������Ƶ
}