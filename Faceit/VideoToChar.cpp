#include "VideoToChar.h"
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

using namespace std;

VideoToChar::VideoToChar(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	//	������ڲ�����Ƶ��Label�Ŀ��
	window_width = ui.label_vedio->width();
	window_heigth = ui.label_vedio->height();
	//	���ö�ʱ��
	frameTimer = new QTimer(this);
}

VideoToChar::~VideoToChar()
{
}

void VideoToChar::mediaInLabel() {

	ui.label_vedio->show();

	//	תΪQImage��ʽ
	QImage image = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format_RGB888);

	QPixmap pixmap = QPixmap::fromImage(image);
	QPixmap fitpixmap = pixmap.scaled(window_width, window_heigth, 
		Qt::KeepAspectRatio, Qt::SmoothTransformation);

	ui.label_vedio->setAlignment(Qt::AlignCenter);
	ui.label_vedio->setPixmap(fitpixmap);
}

//	��һ֡������
void VideoToChar::getNextProcessFrame() {
	
	//	�ж��Ƿ������һ֡
	bool isStop = capture.read(frame);
	if (isStop == false) {
		frameTimer->destroyed();
		return;
	}

	double t = 7;	//	���ű�����СΪ7

	//	����һ����СΪ��ǰ֡����С1/7�Ĵ���֡����¡ͼ
	cv::Mat src = cv::Mat(cv::Size(frame.cols * 1.0 / t, frame.rows * 1.0 / t), src.type());
	cv::Mat src_clone = frame.clone();

	//	����һ���հ�ͼ��
	temp = cv::Mat(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
	cv::resize(frame, src, cv::Size(frame.cols / t, frame.rows / t), 2);

	//	����Ҫת�����ַ�Ϊ0~9������
	string str[10] = { "9","8", "7", "6", "5", "4", "3", "2", "1", "0" };	

	cv::cvtColor(src, src, CV_BGR2GRAY);	
	cv::blur(src, src, cv::Size(3, 3));	//	��ͼƬģ������������ļ��㴦��
	//	��ȡÿһ���ص�bgrֵ
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int c = (int)src.at<uchar>(i, j);
			int b = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[0];
			int g = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[1];
			int r = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[2];
			//	ʹ��putText�������ַ���ӡ���հ�ͼ������ɴ������
			putText(temp, str[(c * 9) / 255], cv::Point(j + j * (t - 1), i + i * (t - 1)), 
				cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(b, g, r), 1, 8);
		}
	}	

	//	��Qt��Label�ؼ�������Ƶ
	mediaInLabel();
}

//	����Ƶ������
void VideoToChar::openVideo() {
	QString srcPath = QFileDialog::getOpenFileName(this,
		"OpenVideo", "../", "Video(*.mp4 *.flv *.mkv *.avi)");
	if (srcPath.isEmpty()) {
		return;
	}

	frameTimer->destroyed();	//	����ʱ������

	capture.open(srcPath.toLocal8Bit().data());

	//	�ж϶�ʱ������״̬
	if (frameTimer->isActive() == false) {
		//	������ʱ��
		frameTimer->start(35);
	}

	//	���Ӵ�����һ֡�Ĳ��ź�
	connect(frameTimer, &QTimer::timeout, this, &VideoToChar::getNextProcessFrame);
}