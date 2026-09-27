cmake_minimum_required(VERSION 3.21)

project(pyTEMInversionGUI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Threads REQUIRED)

add_executable(pytem-inversion-gui
    main.cpp
    MainWindow.cpp
    MainWindow.h
    InversionWorker.cpp
    InversionWorker.h
    MatrixConvolution.cpp
    MatrixConvolution.h
    PlotWidget.cpp
    PlotWidget.h
    TemSolver.cpp
    TemSolver.h
    TransformWeights.h
    UsfReader.cpp
    UsfReader.h
)

target_link_libraries(pytem-inversion-gui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Threads::Threads)
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
        MatrixConvolution.cpp
        TemSolver.cpp
        UsfReader.cpp
    )
    target_include_directories(pytem-native-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(pytem-native-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(pytem-native-tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(pytem-native-tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME pytem-native-tests
             COMMAND pytem-native-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimal.usf)
endif()
