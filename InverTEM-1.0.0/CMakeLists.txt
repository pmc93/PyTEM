cmake_minimum_required(VERSION 3.24)

project(InverTEM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Network)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Network)
find_package(Threads REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)

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
            "Build_InverTEM.bat, for example C:\\Qt\\6.8.0\\msvc2022_64.")
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
        message(STATUS "InverTEM CUDA acceleration enabled")
    else()
        message(STATUS "CUDA toolkit not found; building CPU fallback")
    endif()
endif()

# AVX2 wavenumber-vectorised kernel (explicit intrinsics), selectable in the
# GUI. Only this file is built for AVX2/FMA; TemSolver checks the CPU at run time.
option(PYTEM_ENABLE_VECTOR_KERNEL "Build the AVX2 vectorised forward kernel" ON)
if(PYTEM_ENABLE_VECTOR_KERNEL AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86|x86|i[3-6]86")
    add_compile_definitions(INVERTEM_VECTOR_KERNEL=1)
    if(MSVC)
        set_source_files_properties(src/VectorKernel.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/VectorKernel.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
    message(STATUS "InverTEM AVX2 vectorised kernel enabled")
endif()

add_executable(InverTEM
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/InversionWorker.cpp
    src/InversionWorker.h
    src/JacobianCache.cpp
    src/JacobianCache.h
    src/CudaForward.h
    ${PYTEM_GPU_SOURCE}
    src/ElevationModel.cpp
    src/ElevationModel.h
    src/MapTileLoader.cpp
    src/MapTileLoader.h
    src/MatrixConvolution.cpp
    src/MatrixConvolution.h
    src/Processing.cpp
    src/MainWindowHelpers.h
    src/PlotWidget.cpp
    src/PlotWidget.h
    src/TemSolver.cpp
    src/TemSolver.h
    src/TransformWeights.h
    src/UsfReader.cpp
    src/UsfReader.h
    src/VectorKernel.cpp
    src/VectorKernel.h
)

target_link_libraries(InverTEM PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Network
    Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(InverTEM PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(InverTEM PRIVATE INVERTEM_USE_OPENMP=1)
    message(STATUS "InverTEM OpenMP batch acceleration enabled")
else()
    message(STATUS "OpenMP not found; using the standard C++ thread fallback")
endif()
target_compile_definitions(InverTEM PRIVATE INVERTEM_APP_VERSION="${PROJECT_VERSION}")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/g4.png")
    add_custom_command(TARGET InverTEM POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/g4.png"
            "$<TARGET_FILE_DIR:InverTEM>/g4.png"
        COMMENT "Copying InverTEM icon")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/g4.png" DESTINATION bin)
endif()
if(MSVC)
    target_compile_options(InverTEM PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
    target_compile_options(InverTEM PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()
if(PYTEM_CUDA_ENABLED)
    target_compile_options(InverTEM PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    set_target_properties(InverTEM PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

install(TARGETS InverTEM RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_executable(invertem-native-tests
        tests/native_tests.cpp
        src/JacobianCache.cpp
        src/CudaForward.h
        ${PYTEM_GPU_SOURCE}
        src/MatrixConvolution.cpp
        src/TemSolver.cpp
        src/UsfReader.cpp
        src/VectorKernel.cpp
    )
    target_include_directories(invertem-native-tests PRIVATE src)
    target_link_libraries(invertem-native-tests PRIVATE Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(invertem-native-tests PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(invertem-native-tests PRIVATE INVERTEM_USE_OPENMP=1)
    endif()
    if(MSVC)
        target_compile_options(invertem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    else()
        target_compile_options(invertem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    endif()
    if(PYTEM_CUDA_ENABLED)
        target_compile_options(invertem-native-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        set_target_properties(invertem-native-tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
    add_test(NAME invertem-native-tests
             COMMAND invertem-native-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimal.usf)
endif()
