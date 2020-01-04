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
			QMessageBox::information(this, QStringLiteral("Tip��"), QStringLiteral("����ɹ���"));
		}
	}
}

void PictureMg::reset() {
	loadImage();
}


void PictureMg::heibai() {
	//	�ö�ֵ�����кڰ״���
	cv::threshold(gray, result, 128, 255, cv::THRESH_BINARY);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fanse() {
	//	�������ұ�����ÿһ�����أ�����ȡ��
	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		lut.at<uchar>(i) = 255 - i;
	}

	cv::LUT(result, lut, result);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fudiao() {
	//	ʹ��sobel�˲�����x�����ϵĸ�����
	cv::Sobel(gray, result, CV_8U, 1, 0, 3, 0.4, 128);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::manhua() {
	cv::Mat img;
	//	ʹ��˫���˲���ͼ����н�����ģ��
	cv::bilateralFilter(imageClone, img, 5, 150, 150);
	cv::bilateralFilter(img, imageClone, 5, 150, 150);

	cv::Mat manhua_gray;
	cv::cvtColor(imageClone, manhua_gray, cv::COLOR_BGR2GRAY);

	//	ʹ��������˹���ӻ���ͼ��Ĵ�������
	cv::Mat line_thick;
	cv::Laplacian(gray, line_thick, -1, 3, 1);
	
	//	ʹ��Sobel�˲�����ͼ��Ĵ�������
	cv::Mat line_S;
	cv::Sobel(manhua_gray, line_S, -1, 1, 1);

	cv::Mat line_total;	//	���������
	line_total = line_S + line_thick;

	//	�����һ���޶���Χ�������и�˹�˲�ģ�����������������ֵ��
	cv::normalize(line_total, line_total, 255, 0, CV_MINMAX);
	cv::GaussianBlur(line_total, line_total, cv::Size(3, 3), 3);
	cv::threshold(line_total, line_total, 100, 255, cv::THRESH_BINARY_INV);

	cv::Mat line_total_color;	//	��������Ϊ��ɫ
	cv::cvtColor(line_total, line_total_color, CV_GRAY2BGR);
	//	�����߼�������
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
	//	��������
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			srcX.at<float>(i, j) = j;
			srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
		}
	}

	//	�����Բ�ֵ��ͼ�������ӳ��
	cv::remap(result, result, srcX, srcY, cv::INTER_LINEAR);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::jingxiang() {
	//	flip�������ҶԵ�
	cv::flip(result, result, 1);
	cv::flip(gray, gray, 1);
	cv::flip(imageClone, imageClone, 1);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fugu() {
	//	����ͼ����ͨ��
	cv::Mat b(result.rows, result.cols, CV_32FC1);
	cv::Mat g(result.rows, result.cols, CV_32FC1);
	cv::Mat r(result.rows, result.cols, CV_32FC1);

	std::vector<cv::Mat> channels;
	//	����ͼ��ͨ��
	cv::split(imageClone, channels);
	//	�ı�ͼ��ͨ��ֵ
	channels[0] = 0.272*channels[2] + 0.534*channels[1] + 0.131*channels[0];
	channels[1] = 0.349*channels[2] + 0.686*channels[1] + 0.168*channels[0];
	channels[2] = 0.393*channels[2] + 0.769*channels[1] + 0.189*channels[0];
	//	�ں�ͼ��ͨ��
	cv::merge(channels, result);

	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::miaobian() {
	//	ʹ��canny���Ӷ�ͼ����ȡ�����ﵽ���
	cv::Canny(gray, result, 50, 150);
	cv::imshow(nameWindow, result);

	showMessage();
}

void PictureMg::suoxiao() {
	//	��Сͼ���������ֵ����
	cv::resize(result, result, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
	imshow(nameWindow, result);

	showMessage();
}

void PictureMg::fangda() {
	//	�Ŵ�ͼ��������������ֵЧ�����
	cv::resize(result, result, cv::Size(), 1.5, 1.5, cv::INTER_CUBIC);
	imshow(nameWindow, result);

	showMessage();
}

void PictureMg::shuzi() {
	double t = 7;	//	���ű�����СΪ7

	//	����һ����СΪ��ǰ֡����С1/7�Ĵ���֡����¡ͼ
	cv::Mat src = cv::Mat(cv::Size(result.cols * 1.0 / t, result.rows * 1.0 / t), src.type());
	cv::Mat src_clone = result.clone();

	//	����һ���հ�ͼ��
	cv::Mat temp = cv::Mat(result.size(), result.type(), cv::Scalar(255, 255, 255));
	cv::resize(result, src, cv::Size(result.cols / t, result.rows / t), 2);

	//	����Ҫת�����ַ�Ϊ0~9������
	std::string str[10] = { "9","8", "7", "6", "5", "4", "3", "2", "1", "0" };

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
	imshow(nameWindow, temp);
}

//	����ͼƬ
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

//	����ͼƬ��Ϣ
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