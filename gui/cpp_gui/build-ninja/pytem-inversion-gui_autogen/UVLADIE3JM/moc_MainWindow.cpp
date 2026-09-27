/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.11.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/MainWindow.h"
#include <QtCore/qmetatype.h>

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
        "loadUsf",
        "",
        "selectSounding",
        "index",
        "previousSounding",
        "nextSounding",
        "selectMoment",
        "setLayerCount",
        "count",
        "toggleDataPoint",
        "tableRow",
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
        "exportResult",
        "updateInputPlot"
    };

    QtMocHelpers::UintData qt_methods {
        // Slot 'loadUsf'
        QtMocHelpers::SlotData<void()>(1, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'selectSounding'
        QtMocHelpers::SlotData<void(int)>(3, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 4 },
        }}),
        // Slot 'previousSounding'
        QtMocHelpers::SlotData<void()>(5, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'nextSounding'
        QtMocHelpers::SlotData<void()>(6, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'selectMoment'
        QtMocHelpers::SlotData<void(int)>(7, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 4 },
        }}),
        // Slot 'setLayerCount'
        QtMocHelpers::SlotData<void(int)>(8, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 9 },
        }}),
        // Slot 'toggleDataPoint'
        QtMocHelpers::SlotData<void(int)>(10, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 11 },
        }}),
        // Slot 'toggleSoundingData'
        QtMocHelpers::SlotData<void(int)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'navigateToSounding'
        QtMocHelpers::SlotData<void(int)>(14, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'changeMapBackground'
        QtMocHelpers::SlotData<void()>(15, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'runInversion'
        QtMocHelpers::SlotData<void()>(16, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'killInversion'
        QtMocHelpers::SlotData<void()>(17, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'showProgress'
        QtMocHelpers::SlotData<void(int, int, double, const QString &)>(18, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Int, 20 }, { QMetaType::Double, 21 }, { QMetaType::QString, 22 },
        }}),
        // Slot 'inversionComplete'
        QtMocHelpers::SlotData<void(int, const pytem::InversionResult &)>(23, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { 0x80000000 | 24, 25 },
        }}),
        // Slot 'inversionFailed'
        QtMocHelpers::SlotData<void(int, const QString &)>(26, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::QString, 27 },
        }}),
        // Slot 'batchFinished'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'exportResult'
        QtMocHelpers::SlotData<void()>(29, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'updateInputPlot'
        QtMocHelpers::SlotData<void()>(30, 2, QMC::AccessPrivate, QMetaType::Void),
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
        case 0: _t->loadUsf(); break;
        case 1: _t->selectSounding((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 2: _t->previousSounding(); break;
        case 3: _t->nextSounding(); break;
        case 4: _t->selectMoment((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 5: _t->setLayerCount((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 6: _t->toggleDataPoint((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 7: _t->toggleSoundingData((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 8: _t->navigateToSounding((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 9: _t->changeMapBackground(); break;
        case 10: _t->runInversion(); break;
        case 11: _t->killInversion(); break;
        case 12: _t->showProgress((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 13: _t->inversionComplete((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<pytem::InversionResult>>(_a[2]))); break;
        case 14: _t->inversionFailed((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2]))); break;
        case 15: _t->batchFinished(); break;
        case 16: _t->exportResult(); break;
        case 17: _t->updateInputPlot(); break;
        default: ;
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
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 18)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 18;
    }
    return _id;
}
QT_WARNING_POP
