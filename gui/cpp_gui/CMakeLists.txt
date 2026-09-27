cmake_minimum_required(VERSION 3.24)

project(pyTEMInversionGUI VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Network)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Network)
find_package(Threads REQUIRED)

# A Qt installation built with MinGW cannot be linked into an MSVC/CUDA
# executable. Catch this during configuration instead of failing much later at
# compile/link time with misleading Qt header or unresolved-symbol errors.
if(MSVC)
    if(QT_VERSION_MAJOR EQUAL 6)
        set(PYTEM_QT_PACKAGE_DIR "${Qt6_DIR}")
    else()
        set(PYTEM_QT_PACKAGE_DIR "${Qt5_DIR}")
    endif()
    string(TOLOWER "${PYTEM_QT_PACKAGE_DIR}" PYTEM_QT_PACKAGE_DIR_LOWER)
    if(PYTEM_QT_PACKAGE_DIR_LOWER MATCHES "msys|mingw|ucrt64|win32-g\\+\\+")
        message(FATAL_ERROR
            "The MSVC build found a MinGW/MSYS2 Qt kit at:\n"
            "  ${PYTEM_QT_PACKAGE_DIR}\n"
            "MinGW Qt libraries are not ABI-compatible with MSVC/CUDA. "
            "Install Qt's 'MSVC 2022 64-bit' component and set QT_PREFIX in "
            "Build_pyTEM.bat, for example C:\\Qt\\6.8.0\\msvc2022_64.")
    endif()
endif()

option(PYTEM_ENABLE_CUDA "Build optional NVIDIA CUDA forward acceleration" ON)
set(PYTEM_GPU_SOURCE src/CudaForwardStub.cpp)
set(PYTEM_CUDA_ENABLED OFF)
set(PYTEM_CUDA_ACTIVE OFF CACHE INTERNAL "CUDA acceleration configured for this build tree" FORCE)
if(PYTEM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(PYTEM_GPU_SOURCE src/CudaForward.cu)
        set(PYTEM_CUDA_ENABLED ON)
        set(PYTEM_CUDA_ACTIVE ON CACHE INTERNAL "CUDA acceleration configured for this build tree" FORCE)
        message(STATUS "pyTEM CUDA acceleration enabled")
    else()
        message(STATUS "CUDA toolkit not found; building CPU fallback")
    endif()
endif()

add_executable(pytem-inversion-gui
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/InversionWorker.cpp
    src/InversionWorker.h
    src/JacobianCache.cpp
    src/JacobianCache.h
    src/CudaForward.h
    ${PYTEM_GPU_SOURCE}
    src/MapTileLoader.cpp
    src/MapTileLoader.h
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

target_link_libraries(pytem-inversion-gui PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Network
    Threads::Threads)
target_compile_definitions(pytem-inversion-gui PRIVATE PYTEM_APP_VERSION="${PROJECT_VERSION}")
if(MSVC)
    target_compile_options(pytem-inversion-gui PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
    target_compile_options(pytem-inversion-gui PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()
if(PYTEM_CUDA_ENABLED)
    target_compile_options(pytem-inversion-gui PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    set_target_properties(pytem-inversion-gui PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

install(TARGETS pytem-inversion-gui RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_executable(pytem-native-tests
        tests/native_tests.cpp
        src/JacobianCache.cpp
        src/CudaForward.h
        ${PYTEM_GPU_SOURCE}
        src/MatrixConvolution.cpp
        src/TemSolver.cpp
        src/UsfReader.cpp
    )
    target_include_directories(pytem-native-tests PRIVATE src)
    target_link_libraries(pytem-native-tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(pytem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    else()
        target_compile_options(pytem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    endif()
    if(PYTEM_CUDA_ENABLED)
        target_compile_options(pytem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        set_target_properties(pytem-native-tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
    add_test(NAME pytem-native-tests
             COMMAND pytem-native-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimal.usf)
endif()
