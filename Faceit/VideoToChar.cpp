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

	//	获得用于播放视频的Label的宽高
	window_width = ui.label_vedio->width();
	window_heigth = ui.label_vedio->height();
	//	设置定时器
	frameTimer = new QTimer(this);
}

VideoToChar::~VideoToChar()
{
}

void VideoToChar::mediaInLabel() {

	ui.label_vedio->show();

	//	转为QImage格式
	QImage image = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format_RGB888);

	QPixmap pixmap = QPixmap::fromImage(image);
	QPixmap fitpixmap = pixmap.scaled(window_width, window_heigth, 
		Qt::KeepAspectRatio, Qt::SmoothTransformation);

	ui.label_vedio->setAlignment(Qt::AlignCenter);
	ui.label_vedio->setPixmap(fitpixmap);
}

//	下一帧处理函数
void VideoToChar::getNextProcessFrame() {
	
	//	判断是否存在下一帧
	bool isStop = capture.read(frame);
	if (isStop == false) {
		frameTimer->destroyed();
		return;
	}

	double t = 7;	//	缩放比例大小为7

	//	创建一幅大小为当前帧数大小1/7的处理帧及克隆图
	cv::Mat src = cv::Mat(cv::Size(frame.cols * 1.0 / t, frame.rows * 1.0 / t), src.type());
	cv::Mat src_clone = frame.clone();

	//	创建一幅空白图像
	temp = cv::Mat(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
	cv::resize(frame, src, cv::Size(frame.cols / t, frame.rows / t), 2);

	//	定义要转换的字符为0~9的数字
	string str[10] = { "9","8", "7", "6", "5", "4", "3", "2", "1", "0" };	

	cv::cvtColor(src, src, CV_BGR2GRAY);	
	cv::blur(src, src, cv::Size(3, 3));	//	将图片模糊化方便下面的计算处理
	//	获取每一像素的bgr值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int c = (int)src.at<uchar>(i, j);
			int b = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[0];
			int g = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[1];
			int r = (int)src_clone.at<cv::Vec3b>(i + i * (t - 1), j + j * (t - 1))[2];
			//	使用putText函数将字符打印到空白图像中完成处理操作
			putText(temp, str[(c * 9) / 255], cv::Point(j + j * (t - 1), i + i * (t - 1)), 
				cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(b, g, r), 1, 8);
		}
	}	

	//	在Qt的Label控件播放视频
	mediaInLabel();
}

//	打开视频并处理
void VideoToChar::openVideo() {
	QString srcPath = QFileDialog::getOpenFileName(this,
		"OpenVideo", "../", "Video(*.mp4 *.flv *.mkv *.avi)");
	if (srcPath.isEmpty()) {
		return;
	}

	frameTimer->destroyed();	//	将定时器销毁

	capture.open(srcPath.toLocal8Bit().data());

	//	判断定时器激活状态
	if (frameTimer->isActive() == false) {
		//	启动定时器
		frameTimer->start(35);
	}

	//	连接处理下一帧的槽信号
	connect(frameTimer, &QTimer::timeout, this, &VideoToChar::getNextProcessFrame);
}