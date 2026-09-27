#include "MainWindow.h"

#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QStyleFactory>
#include <QString>

#ifndef INVERTEM_APP_VERSION
#define INVERTEM_APP_VERSION "development"
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("InverTEM");
    QApplication::setApplicationVersion(INVERTEM_APP_VERSION);
    QApplication::setOrganizationName("InverTEM");
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    const QString iconPath = QDir(QApplication::applicationDirPath())
        .filePath("g4.png");
    if (!QIcon(iconPath).isNull())
        QApplication::setWindowIcon(QIcon(iconPath));

    MainWindow window;
    window.show();
    return app.exec();
}
