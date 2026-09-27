#include "MainWindow.h"

#include <QApplication>
#include <QStyleFactory>
#include <QString>

#ifndef PYTEM_APP_VERSION
#define PYTEM_APP_VERSION "development"
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("pyTEM Inversion");
    QApplication::setApplicationVersion(PYTEM_APP_VERSION);
    QApplication::setOrganizationName("pyTEM");
    QApplication::setStyle(QStyleFactory::create("Fusion"));

    MainWindow window;
    window.setWindowTitle(QString("pyTEM Native C++ Inversion v%1 — parallel USF batch")
                              .arg(QApplication::applicationVersion()));
    window.show();
    return app.exec();
}
