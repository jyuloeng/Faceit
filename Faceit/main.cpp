#include "Faceit.h"
#include <QtWidgets/QApplication>
#include "qfile.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Faceit w;
	w.show();
	return a.exec();
}
