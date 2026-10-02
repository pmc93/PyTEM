/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.11.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/MainWindow.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.11.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN10MainWindowE_t {};
} // unnamed namespace

template <> constexpr inline auto MainWindow::qt_create_metaobjectdata<qt_meta_tag_ZN10MainWindowE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "MainWindow",
        "newProject",
        "",
        "saveProject",
        "loadProject",
        "loadUsf",
        "correctElevations",
        "selectSounding",
        "index",
        "previousSounding",
        "nextSounding",
        "selectMoment",
        "setLayerCount",
        "count",
        "toggleDataPoint",
        "tableRow",
        "editDataPoints",
        "QList<int>",
        "pointIds",
        "restore",
        "undoGateEdit",
        "selectPlotMode",
        "previousTransectPage",
        "nextTransectPage",
        "toggleSoundingData",
        "soundingIndex",
        "navigateToSounding",
        "changeMapBackground",
        "runInversion",
        "killInversion",
        "showProgress",
        "jobIndex",
        "iteration",
        "rms",
        "detail",
        "inversionComplete",
        "pytem::InversionResult",
        "result",
        "inversionFailed",
        "message",
        "batchFinished",
        "exportModelledData",
        "clearSavedResults",
        "moveKeptUsfFiles",
        "updateInputPlot"
    };

    QtMocHelpers::UintData qt_methods {
        // Slot 'newProject'
        QtMocHelpers::SlotData<void()>(1, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'saveProject'
        QtMocHelpers::SlotData<void()>(3, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'loadProject'
        QtMocHelpers::SlotData<void()>(4, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'loadUsf'
        QtMocHelpers::SlotData<void()>(5, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'correctElevations'
        QtMocHelpers::SlotData<void()>(6, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'selectSounding'
        QtMocHelpers::SlotData<void(int)>(7, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 8 },
        }}),
        // Slot 'previousSounding'
        QtMocHelpers::SlotData<void()>(9, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'nextSounding'
        QtMocHelpers::SlotData<void()>(10, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'selectMoment'
        QtMocHelpers::SlotData<void(int)>(11, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 8 },
        }}),
        // Slot 'setLayerCount'
        QtMocHelpers::SlotData<void(int)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'toggleDataPoint'
        QtMocHelpers::SlotData<void(int)>(14, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 15 },
        }}),
        // Slot 'editDataPoints'
        QtMocHelpers::SlotData<void(const QVector<int> &, bool)>(16, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 17, 18 }, { QMetaType::Bool, 19 },
        }}),
        // Slot 'undoGateEdit'
        QtMocHelpers::SlotData<void()>(20, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'selectPlotMode'
        QtMocHelpers::SlotData<void(int)>(21, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 8 },
        }}),
        // Slot 'previousTransectPage'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'nextTransectPage'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'toggleSoundingData'
        QtMocHelpers::SlotData<void(int)>(24, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 25 },
        }}),
        // Slot 'navigateToSounding'
        QtMocHelpers::SlotData<void(int)>(26, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 25 },
        }}),
        // Slot 'changeMapBackground'
        QtMocHelpers::SlotData<void()>(27, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runInversion'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'killInversion'
        QtMocHelpers::SlotData<void()>(29, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'showProgress'
        QtMocHelpers::SlotData<void(int, int, double, const QString &)>(30, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 31 }, { QMetaType::Int, 32 }, { QMetaType::Double, 33 }, { QMetaType::QString, 34 },
        }}),
        // Slot 'inversionComplete'
        QtMocHelpers::SlotData<void(int, const pytem::InversionResult &)>(35, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 31 }, { 0x80000000 | 36, 37 },
        }}),
        // Slot 'inversionFailed'
        QtMocHelpers::SlotData<void(int, const QString &)>(38, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 31 }, { QMetaType::QString, 39 },
        }}),
        // Slot 'batchFinished'
        QtMocHelpers::SlotData<void()>(40, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'exportModelledData'
        QtMocHelpers::SlotData<void()>(41, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'clearSavedResults'
        QtMocHelpers::SlotData<void()>(42, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'moveKeptUsfFiles'
        QtMocHelpers::SlotData<void()>(43, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'updateInputPlot'
        QtMocHelpers::SlotData<void()>(44, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<MainWindow, qt_meta_tag_ZN10MainWindowE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN10MainWindowE_t>.metaTypes,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MainWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->newProject(); break;
        case 1: _t->saveProject(); break;
        case 2: _t->loadProject(); break;
        case 3: _t->loadUsf(); break;
        case 4: _t->correctElevations(); break;
        case 5: _t->selectSounding((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 6: _t->previousSounding(); break;
        case 7: _t->nextSounding(); break;
        case 8: _t->selectMoment((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 9: _t->setLayerCount((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 10: _t->toggleDataPoint((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 11: _t->editDataPoints((*reinterpret_cast<std::add_pointer_t<QList<int>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[2]))); break;
        case 12: _t->undoGateEdit(); break;
        case 13: _t->selectPlotMode((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 14: _t->previousTransectPage(); break;
        case 15: _t->nextTransectPage(); break;
        case 16: _t->toggleSoundingData((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 17: _t->navigateToSounding((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 18: _t->changeMapBackground(); break;
        case 19: _t->runInversion(); break;
        case 20: _t->killInversion(); break;
        case 21: _t->showProgress((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 22: _t->inversionComplete((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<pytem::InversionResult>>(_a[2]))); break;
        case 23: _t->inversionFailed((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2]))); break;
        case 24: _t->batchFinished(); break;
        case 25: _t->exportModelledData(); break;
        case 26: _t->clearSavedResults(); break;
        case 27: _t->moveKeptUsfFiles(); break;
        case 28: _t->updateInputPlot(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<int> >(); break;
            }
            break;
        }
    }
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.strings))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 29)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 29;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 29)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 29;
    }
    return _id;
}
QT_WARNING_POP
