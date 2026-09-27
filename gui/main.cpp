#include "MainWindow.h"

#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("pyTEM Inversion");
    QApplication::setOrganizationName("pyTEM");
    QApplication::setStyle(QStyleFactory::create("Fusion"));

    MainWindow window;
    window.show();
    return app.exec();
}

