#include "PictureMg.h"
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

PictureMg::PictureMg(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
}

PictureMg::~PictureMg()
{
}

void PictureMg::open() {
	path = QFileDialog::getOpenFileName(this,
		QStringLiteral("OpenImage", "../", "Image(*.png *jpg *bmp)"));

	loadImage();
}

void PictureMg::save() {
	QString path = QFileDialog::getSaveFileName(this,
		QStringLiteral("SaveImage", "../", "Image(*.png *jpg *bmp)"));

	if (path.isEmpty()) {
		return;
	}
	else {
		if (cv::imwrite(path.toLocal8Bit().data(), result)) {
			QMessageBox::information(this, QStringLiteral("Tip！"), QStringLiteral("保存成功！"));
		}
	}
}

void PictureMg::reset() {
	loadImage();
}


void PictureMg::heibai() {
	//	用二值化进行黑白处理
	cv::threshold(gray, result, 128, 255, cv::THRESH_BINARY);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fanse() {
	//	建立查找表，访问每一个像素，对齐取反
	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		lut.at<uchar>(i) = 255 - i;
	}

	cv::LUT(result, lut, result);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fudiao() {
	//	使用sobel滤波进行x方向上的浮雕处理
	cv::Sobel(gray, result, CV_8U, 1, 0, 3, 0.4, 128);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::manhua() {
	cv::Mat img;
	//	使用双边滤波对图像进行降噪与模糊
	cv::bilateralFilter(imageClone, img, 5, 150, 150);
	cv::bilateralFilter(img, imageClone, 5, 150, 150);

	cv::Mat manhua_gray;
	cv::cvtColor(imageClone, manhua_gray, cv::COLOR_BGR2GRAY);

	//	使用拉普拉斯算子画出图像的粗线轮廓
	cv::Mat line_thick;
	cv::Laplacian(gray, line_thick, -1, 3, 1);
	
	//	使用Sobel滤波画出图像的粗线轮廓
	cv::Mat line_S;
	cv::Sobel(manhua_gray, line_S, -1, 1, 1);

	cv::Mat line_total;	//	获得总轮廓
	line_total = line_S + line_thick;

	//	矩阵归一化限定范围，并进行高斯滤波模糊轮廓，最后将轮廓二值化
	cv::normalize(line_total, line_total, 255, 0, CV_MINMAX);
	cv::GaussianBlur(line_total, line_total, cv::Size(3, 3), 3);
	cv::threshold(line_total, line_total, 100, 255, cv::THRESH_BINARY_INV);

	cv::Mat line_total_color;	//	将轮廓变为彩色
	cv::cvtColor(line_total, line_total_color, CV_GRAY2BGR);
	//	进行逻辑与运算
	cv::bitwise_and(imageClone, line_total_color, result);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::sumiao() {

	cv::Mat gray1;
	cvtColor(image, gray1, cv::COLOR_BGR2GRAY);

	cv::Mat sobelX, sobelY;
	Sobel(gray1, sobelX, CV_8U, 1, 0, 3, 0.4, 128);	Sobel(gray1, sobelY, CV_8U, 0, 1, 3, 0.4, 128);	Sobel(gray1, sobelX, CV_16S, 1, 0);	Sobel(gray1, sobelY, CV_16S, 0, 1);	cv::Mat sobel;	sobel = abs(sobelX) + abs(sobelY);	double sobmin, sobmax;	cv::minMaxLoc(sobel, &sobmin, &sobmax);	cv::Mat result;	sobel.convertTo(result, CV_8U, -255. / sobmax, 255);	cv::imshow(nameWindow, result);

	showMessage();

}

void PictureMg::bolang() {
	cv::Mat	srcX(result.rows, result.cols, CV_32F);
	cv::Mat	srcY(result.rows, result.cols, CV_32F);
	//	调整像素
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			srcX.at<float>(i, j) = j;
			srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
		}
	}

	//	用线性插值对图像进行重映射
	cv::remap(result, result, srcX, srcY, cv::INTER_LINEAR);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::jingxiang() {
	//	flip函数左右对调
	cv::flip(result, result, 1);
	cv::flip(gray, gray, 1);
	cv::flip(imageClone, imageClone, 1);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fugu() {
	//	定义图像三通道
	cv::Mat b(result.rows, result.cols, CV_32FC1);
	cv::Mat g(result.rows, result.cols, CV_32FC1);
	cv::Mat r(result.rows, result.cols, CV_32FC1);

	std::vector<cv::Mat> channels;
	//	分离图像通道
	cv::split(imageClone, channels);
	//	改变图像通道值
	channels[0] = 0.272*channels[2] + 0.534*channels[1] + 0.131*channels[0];
	channels[1] = 0.349*channels[2] + 0.686*channels[1] + 0.168*channels[0];
	channels[2] = 0.393*channels[2] + 0.769*channels[1] + 0.189*channels[0];
	//	融合图像通道
	cv::merge(channels, result);

	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::miaobian() {
	//	使用canny算子对图像提取轮廓达到描边
	cv::Canny(gray, result, 50, 150);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::suoxiao() {
	//	缩小图像用区域插值即可
	cv::resize(result, result, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
	imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fangda() {
	//	放大图像用三次样条插值效果最好
	cv::resize(result, result, cv::Size(), 1.5, 1.5, cv::INTER_CUBIC);
	imshow(nameWindow, result);

	showMessage();
}

void PictureMg::shuzi() {
	double t = 7;	//	缩放比例大小为7

	//	创建一幅大小为当前帧数大小1/7的处理帧及克隆图
	cv::Mat src = cv::Mat(cv::Size(result.cols * 1.0 / t, result.rows * 1.0 / t), src.type());
	cv::Mat src_clone = result.clone();

	//	创建一幅空白图像
	cv::Mat temp = cv::Mat(result.size(), result.type(), cv::Scalar(255, 255, 255));
	cv::resize(result, src, cv::Size(result.cols / t, result.rows / t), 2);

	//	定义要转换的字符为0~9的数字
	std::string str[10] = { "9","8", "7", "6", "5", "4", "3", "2", "1", "0" };

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
	imshow(nameWindow, temp);
}

//	加载图片
void PictureMg::loadImage() {
	if (path.isEmpty()) {
		return;
	}
	image = cv::imread(path.toLocal8Bit().data());

	cv::imshow(nameWindow, image);
	imageClone = image.clone();
	cv::cvtColor(imageClone, gray, cv::COLOR_BGR2GRAY);
	result = image;

	showMessage();
}

//	加载图片信息
void PictureMg::showMessage() {
	int cols = result.cols;
	int rows = result.rows;
	int channels = result.channels();
	long pixes = cols * rows*channels;
	
	ui.label_cols->setText(QString::number(cols));
	ui.label_rows->setText(QString::number(rows));
	ui.label_channels->setText(QString::number(channels));
	ui.label_pix->setText(QString::number(pixes));
}