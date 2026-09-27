cmake_minimum_required(VERSION 3.21)

project(pyTEMInversionGUI VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Threads REQUIRED)

add_executable(pytem-inversion-gui
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/InversionWorker.cpp
    src/InversionWorker.h
    src/JacobianCache.cpp
    src/JacobianCache.h
    src/MatrixConvolution.cpp
    src/MatrixConvolution.h
    src/PlotWidget.cpp
    src/PlotWidget.h
    src/TemSolver.cpp
    src/TemSolver.h
    src/TransformWeights.h
    src/UsfReader.cpp
    src/UsfReader.h
)

target_link_libraries(pytem-inversion-gui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Threads::Threads)
target_compile_definitions(pytem-inversion-gui PRIVATE PYTEM_APP_VERSION="${PROJECT_VERSION}")
if(MSVC)
    target_compile_options(pytem-inversion-gui PRIVATE /W4 /permissive-)
else()
    target_compile_options(pytem-inversion-gui PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS pytem-inversion-gui RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_executable(pytem-native-tests
        tests/native_tests.cpp
        src/JacobianCache.cpp
        src/MatrixConvolution.cpp
        src/TemSolver.cpp
        src/UsfReader.cpp
    )
    target_include_directories(pytem-native-tests PRIVATE src)
    target_link_libraries(pytem-native-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(pytem-native-tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(pytem-native-tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME pytem-native-tests
             COMMAND pytem-native-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimal.usf)
endif()
