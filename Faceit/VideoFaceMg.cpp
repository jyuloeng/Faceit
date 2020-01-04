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

	//	加载人脸检测的级联分类器
	faceDetector.load("haarcascade_frontalface_alt2.xml");

	//	加载dlib人脸特征点检测器
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//	获得用于播放视频的Label的宽高
	window_heigth = ui.label_media->height();	
	window_width = ui.label_media->width();

	frameTimer = new QTimer(this);	//	初始化定时器
}

VideoFaceMg::~VideoFaceMg()
{
}

void VideoFaceMg::openVideo() {
	isCameraOrVideoFlag = false;	//	false代表为视频文件

	QString srcPath = QFileDialog::getOpenFileName(this,
		"打开视频", "../", "Video(*.mp4 *.flv *.mkv *.avi)");
	if (srcPath.isEmpty()) {
		return;
	}
	//	显示在label上
	ui.label_media->show();
	capture.open(srcPath.toLocal8Bit().data());

	frameTimer->destroyed();	//	将定时器销毁

	//	判断定时器激活状态
	if (frameTimer->isActive() == false) {
		//	启动定时器
		frameTimer->start(35);
	}
	//	连接处理下一帧的函数槽信号
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getNextFrame);
}

void VideoFaceMg::closeVideo() {
	capture.release();	//	释放资源
	frameTimer->stop();	//	停止定时器

	ui.label_media->close();
}

void VideoFaceMg::openCamera() {
	isCameraOrVideoFlag = true;

	ui.label_media->show();
	capture.open(0);

	frameTimer->destroyed();

	//	判断定时器激活状态
	if (frameTimer->isActive() == false) {
		//	启动定时器
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
	//	将Mat型temp转为Qt的QImage型
	screenImage = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format::Format_RGB888);

	ui.label_printScreen->setPixmap(QPixmap::fromImage(screenImage));
	ui.label_printScreen->setScaledContents(true);

	//	保存到本地
	const QPixmap *screenShot = ui.label_printScreen->pixmap();
	QString path = QFileDialog::getSaveFileName(this, "save", "../", "Image(*.jpg)");
	if (path.isEmpty()) {
		return;
	}
	else {
		if (screenShot->save(path)) {
			QMessageBox::information(this, 
				QStringLiteral("Tip！"), QStringLiteral("截图保存成功！"));
		}
	}
}

void VideoFaceMg::recordVideo() {
	double rate = 6.0;	//	设置录制帧数
	cv::Size videoSize(frame.cols, frame.rows);	//	设置大小

	//	开启写视频文件对象
	writer.open("RecordVideo.mp4", 
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), rate, videoSize, true);
	isRecord = true;

	while (isRecord) {
		capture.read(frame);

		//	镜像操作
		cv::flip(frame, frame, 1);
		writer.write(frame);	//	写视频文件
		cv::imshow("录制中...", frame);
		cv::waitKey(35);
	}
}

void VideoFaceMg::finishRecord() {
	cv::destroyWindow("录制中...");	//	关闭窗口
	isRecord = false;
	writer.release();	//	释放资源
}

void VideoFaceMg::faceRecognition() {
	//	连接获取下一人脸识别处理帧槽信号
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getProcessFrame);
}

//	输入待测人脸
void VideoFaceMg::faceSwapInput() {
	//	打开并设置图片
	QString path = QFileDialog::getOpenFileName(this,
		QStringLiteral("OpenImage1", "../", "Image(*.png *jpg *bmp)"));
	if (path.isEmpty()) {
		return;
	}

	faceSrc = cv::imread(path.toLocal8Bit().data());

	//	将图像进行一系列处理已提取较好的面容
	cv::cvtColor(faceSrc, faceGray, CV_BGR2GRAY);	
	cv::equalizeHist(faceGray, faceGray);	//	直方图均衡提升对比度
	cv::medianBlur(faceGray, faceGray, 3);	//	中值滤波轻微降噪

	QPixmap pixmap = QPixmap(path).scaled(ui.label_faceSwapInput->width(), ui.label_faceSwapInput->height(),
		Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.label_faceSwapInput->setAlignment(Qt::AlignCenter);
	ui.label_faceSwapInput->setPixmap(pixmap);
}

//	人脸交换
void VideoFaceMg::faceSwap() {
	//	连接获取下一人脸交换处理帧槽信号
	connect(frameTimer, &QTimer::timeout, this, &VideoFaceMg::getFaceSwapFrame);
}

//	在label处播放视频
void VideoFaceMg::mediaToLabel() {
	if (isCameraOrVideoFlag == true) {
		cv::flip(frame, frame, 1);
	}

	cv::cvtColor(frame, temp, CV_BGR2RGB);

	//	转为QImage格式
	QImage image = QImage((const unsigned char *)(temp.data), temp.cols,
		temp.rows, QImage::Format_RGB888);

	QPixmap pixmap = QPixmap::fromImage(image);
	QPixmap fitpixmap = pixmap.scaled(window_width, window_heigth, Qt::KeepAspectRatio, Qt::SmoothTransformation);

	ui.label_media->setAlignment(Qt::AlignCenter);
	ui.label_media->setPixmap(fitpixmap);
}

//	人脸识别帧处理函数
void VideoFaceMg::frameProcess(cv::Mat &output) {

	cv::Mat gray;	
	cv::cvtColor(output, gray, cv::COLOR_BGR2GRAY);	
	cv::equalizeHist(gray, gray);	//	直方图均衡化

	//	Mat类型转化为dlib的array2d类型
	dlib::array2d<dlib::bgr_pixel> dlib_img;
	dlib::assign_image(dlib_img, dlib::cv_image<uchar>(gray));

	std::vector<cv::Rect> cv_faces;			//	opencv库类型的脸部区域
	std::vector<dlib::rectangle> dlib_faces;//	  dlib库类型的脸部区域

	//	haar级联分类器探测人脸区域，获取一系列人脸所在区域
	faceDetector.detectMultiScale(gray, cv_faces);

	for (int i = 0; i < cv_faces.size(); i++)
	{
		cv::rectangle(output, cv_faces[i], CV_RGB(200, 0, 0));
		dlib::rectangle rect(cv_faces[i].x, cv_faces[i].y,
			cv_faces[i].x + cv_faces[i].width, cv_faces[i].y + cv_faces[i].height);
		dlib_faces.push_back(rect);
	}

	//	获取人脸68个特征点部位分布(拿到每一张脸保存在shapes中)
	std::vector<dlib::full_object_detection> shapes;
	for (int i = 0; i < dlib_faces.size(); i++)
	{
		dlib::full_object_detection shape = sp(dlib_img, dlib_faces[i]);
		shapes.push_back(shape);
	}

	//	画出轮廓
	faceLine(output, shapes);
}

//	人脸交换帧处理函数
void VideoFaceMg::faceSwapProcess(cv::Mat &output) {
	//	先检测图二人脸，因为要将图一提取的roi变成图二的大小	（图二为视频当前帧）
	faceDetector.detectMultiScale(output, face2);		//	检测图二人脸
	if (face2.size() == 0) {}
	else {	//	若图二存在人脸
		for (int i = 0; i < face2.size(); i++)
		{
			roi2 = output(face2[i]);	//	获得图二人脸区域
			//	获得图二人脸中心坐标
			face2_center_x = cvRound((face2[i].x + face2[i].width * 0.5));
			face2_center_y = cvRound((face2[i].y + face2[i].height * 0.5));
		}
		
		faceDetector.detectMultiScale(faceGray, face1);	//	检测图一人脸
		if (face1.size() == 0) {
			QMessageBox::information(this, QStringLiteral("Warring！"), 
				QStringLiteral("待测图未能识别出人脸，请重新选择！"));
			return;
		}
		for (int i = 0; i < face1.size(); i++)
		{
			roi = faceSrc(face1[i]);	//	获得图一人脸区域并调整大小
			cv::resize(roi, roi, cv::Size(roi2.cols*0.95, roi2.rows*0.95));
		}

		cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8U);	//	创建掩码

		cv::Point face1_center(mask.cols / 2, mask.rows / 2);
		//	face1的面部大小半径
		int radius = std::min(mask.cols, mask.rows) / 2.3;

		cv::ellipse(mask, face1_center,		//	使用椭圆画出掩码
			cv::Size(radius*0.9, radius*1.05), 0, -180, 180, cv::Scalar(255), -1);

		roi.copyTo(roi, mask);

		cv::Point face2_center(face2_center_x, face2_center_y);

		//	在相应位置进行松柏融合
		seamlessClone(roi, output, mask, face2_center, output, cv::NORMAL_CLONE);
	}
}

//	画出轮廓
void VideoFaceMg::faceLine(cv::Mat img, std::vector<dlib::full_object_detection> faces)
{
	int i, j;
	for (j = 0; j < faces.size(); j++)
	{
		cv::Point startPoint, endPoint;	//	轮廓起点与终点

		for (i = 0; i < 67; i++)
		{
			switch (i)
			{
			case 16:	//下巴到脸颊 0 ~ 16
			case 21:	//左边眉毛	17 ~ 21
			case 26:	//右边眉毛	21 ~ 26
			case 30:	//鼻梁		27 ~ 30
			case 35:	//鼻孔		31 ~ 35
			case 41:	//左眼		36 ~ 41
			case 47:	//右眼		42 ~ 47
			case 59:	//嘴唇外圈	48 ~ 59 嘴唇内圈	59 ~ 67
				i++;
				break;
			default:
				break;
			}

			startPoint.x = faces[j].part(i).x();
			startPoint.y = faces[j].part(i).y();
			endPoint.x = faces[j].part(i + 1).x();
			endPoint.y = faces[j].part(i + 1).y();
			//	line函数进行画线
			cv::line(img, startPoint, endPoint, cv::Scalar(0, 0, 255), 1);
		}
	}
}

//	获取下一帧
void VideoFaceMg::getNextFrame() {
	//	判断是否有下一帧
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	//	在label处播放视频
	mediaToLabel();
}

//	获得下一人脸识别处理帧
void VideoFaceMg::getProcessFrame() {
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	frameProcess(frame);	//	人脸识别帧处理函数
	mediaToLabel();			//	在label处播放视频
}

//	获得下一人脸交换处理帧
void VideoFaceMg::getFaceSwapFrame() {
	bool isPlay = capture.read(frame);

	if (isPlay == false) {
		closeCamera();
		return;
	}

	faceSwapProcess(frame);	//	人脸交换帧处理函数
	mediaToLabel();			//	在label处播放视频
}