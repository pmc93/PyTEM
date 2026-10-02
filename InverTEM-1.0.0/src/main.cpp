#include "MainWindow.h"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QIcon>
#include <QScreen>
#include <QStyleFactory>
#include <QString>
#include <QSysInfo>

#include <cstdio>
#include <exception>

#ifdef _WIN32
#include <share.h>
#include <windows.h>
#endif

#ifndef INVERTEM_APP_VERSION
#define INVERTEM_APP_VERSION "development"
#endif

// invertem_log.txt (next to the exe, else in %TEMP%) records start-up steps,
// every Qt message and any crash, flushed per line so it survives a crash.
static FILE *logFile = nullptr;

static void writeLog(const char *text)
{
    if (!logFile) return;
    std::fprintf(logFile, "%s %s\n",
                 QDateTime::currentDateTime().toString("HH:mm:ss.zzz").toUtf8().constData(), text);
    std::fflush(logFile);
}

static void messageHandler(QtMsgType type, const QMessageLogContext &, const QString &message)
{
    static const char *const names[] = {"debug", "warning", "critical", "fatal", "info"};
    writeLog(QString("[%1] %2").arg(names[type], message).toUtf8().constData());
}

#ifdef _WIN32
static LONG WINAPI crashHandler(EXCEPTION_POINTERS *info)
{
    const auto *record = info->ExceptionRecord;
    HMODULE module = nullptr;
    char name[MAX_PATH] = "unknown module";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(record->ExceptionAddress), &module))
        GetModuleFileNameA(module, name, MAX_PATH);
    char text[MAX_PATH + 128];
    std::snprintf(text, sizeof(text), "[crash] exception 0x%08lX in %s at offset 0x%llX",
                  record->ExceptionCode, name,
                  static_cast<unsigned long long>(reinterpret_cast<const char *>(record->ExceptionAddress)
                                                  - reinterpret_cast<const char *>(module)));
    writeLog(text);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int main(int argc, char *argv[])
{
    // The folder the user launched the exe from. The single-file exe runs from a
    // temporary copy, so the module path is only a fallback to the command line.
#ifdef _WIN32
    int count = 0;
    wchar_t **arguments = CommandLineToArgvW(GetCommandLineW(), &count);
    QString launched = count > 0 ? QString::fromWCharArray(arguments[0]) : QString();
    LocalFree(arguments);
    if (!launched.endsWith(".exe", Qt::CaseInsensitive)) launched += ".exe";
    if (!QFileInfo::exists(launched)) {
        wchar_t modulePath[MAX_PATH];
        GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
        launched = QString::fromWCharArray(modulePath);
    }
    SetUnhandledExceptionFilter(crashHandler);
    qputenv("QT_DEBUG_PLUGINS", "1"); // logs how the Qt plugins (platforms/qwindows.dll) load
#else
    const QString launched = QString::fromLocal8Bit(argv[0]);
#endif
    const QString exeDir = QFileInfo(launched).absolutePath();
    for (const QString &dir : {exeDir, QDir::tempPath()}) {
        const QString path = QDir::toNativeSeparators(QDir(dir).filePath("invertem_log.txt"));
#ifdef _WIN32
        logFile = _wfsopen(reinterpret_cast<const wchar_t *>(path.utf16()), L"w", _SH_DENYNO); // readable while running
#else
        logFile = std::fopen(path.toLocal8Bit().constData(), "w");
#endif
        if (logFile) break;
    }
    qInstallMessageHandler(messageHandler);
    std::set_terminate([] { writeLog("[crash] unhandled C++ exception"); std::abort(); });
    qInfo() << "InverTEM" << INVERTEM_APP_VERSION << "starting in" << exeDir;
    qInfo() << "System:" << QSysInfo::prettyProductName() << QSysInfo::kernelVersion()
            << QSysInfo::currentCpuArchitecture() << "Qt" << qVersion();
#ifdef _WIN32
    qInfo() << "AVX2:" << bool(IsProcessorFeaturePresent(40 /* PF_AVX2_INSTRUCTIONS_AVAILABLE */));
#endif

    QApplication app(argc, argv);
    app.setProperty("exeDir", exeDir); // where invertem_solver.txt is kept
    qputenv("QT_DEBUG_PLUGINS", "0");
    qInfo() << "QApplication created, platform" << QGuiApplication::platformName();
    for (const QScreen *screen : QGuiApplication::screens())
        qInfo() << "Screen" << screen->name() << screen->geometry() << "scale" << screen->devicePixelRatio();
    QApplication::setApplicationName("InverTEM");
    QApplication::setApplicationVersion(INVERTEM_APP_VERSION);
    QApplication::setOrganizationName("InverTEM");
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    const QString iconPath = QDir(QApplication::applicationDirPath()).filePath("g4.png");
    if (!QIcon(iconPath).isNull())
        QApplication::setWindowIcon(QIcon(iconPath));

    MainWindow window;
    window.showMaximized();
    qInfo() << "Main window shown";
    const int result = app.exec();
    qInfo() << "Exited with code" << result;
    return result;
}
