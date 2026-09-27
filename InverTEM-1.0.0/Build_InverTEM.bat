@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
set "BUILD_DIR=build-ninja"
set "PARALLEL_JOBS=4"
set "ENABLE_CUDA=ON"
set "CUDA_NVCC="
set "VS_INSTALL="
set "MSVC_CL="
set "CMAKE_FRESH_ARG="
set "APP_EXE="
set "APP_DIR="

rem If CMake cannot find Qt automatically, put its kit directory here, e.g.:
rem set "QT_PREFIX=C:\Qt\6.8.0\msvc2022_64"
set "QT_PREFIX="

cd /d "%~dp0"
if errorlevel 1 goto location_failed

echo Project directory: %CD%
if not exist "CMakeLists.txt" goto missing_project
if not exist "%CMAKE_EXE%" goto missing_cmake

if /I "%ENABLE_CUDA%"=="ON" call :find_cuda
if defined CUDA_NVCC (
    echo CUDA compiler: !CUDA_NVCC!
    call :find_msvc
    if errorlevel 1 goto missing_msvc
    if not defined QT_PREFIX call :find_qt_msvc
    if defined QT_PREFIX echo Qt MSVC kit: !QT_PREFIX!
) else if /I "%ENABLE_CUDA%"=="ON" (
    echo WARNING: nvcc.exe was not found. The NVIDIA display driver alone is not the CUDA toolkit.
    echo Install the CUDA Toolkit, or make sure CUDA_PATH points to its installation folder.
)

rem Do not retain a MinGW/MSYS2 Qt selection after switching to MSVC for CUDA.
if defined MSVC_CL if exist "%BUILD_DIR%\CMakeCache.txt" (
    findstr /I /R /C:"^Qt6_DIR.*msys64" /C:"^Qt6_DIR.*mingw" /C:"^Qt6_DIR.*ucrt64" "%BUILD_DIR%\CMakeCache.txt" >nul
    if not errorlevel 1 (
        echo Refreshing the previous MinGW Qt configuration...
        set "CMAKE_FRESH_ARG=--fresh"
    )
)

rem A previous CPU-only or failed CUDA cache must be regenerated under MSVC.
if defined CUDA_NVCC if exist "%BUILD_DIR%\CMakeCache.txt" (
    findstr /L /C:"PYTEM_CUDA_ACTIVE:INTERNAL=ON" "%BUILD_DIR%\CMakeCache.txt" >nul
    if errorlevel 1 (
        echo Refreshing the previous CPU-only or incomplete CUDA configuration...
        set "CMAKE_FRESH_ARG=--fresh"
    )
)

rem Configure on every run so changes to CUDA, Qt, or build options are detected.
echo.
echo Configuring InverTEM 1.0.0...
if defined CUDA_NVCC (
    if defined QT_PREFIX (
        "%CMAKE_EXE%" !CMAKE_FRESH_ARG! -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTEM_ENABLE_CUDA=%ENABLE_CUDA% "-DCMAKE_CUDA_COMPILER:FILEPATH=!CUDA_NVCC!" -DCMAKE_PREFIX_PATH="%QT_PREFIX%"
    ) else (
        "%CMAKE_EXE%" !CMAKE_FRESH_ARG! -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTEM_ENABLE_CUDA=%ENABLE_CUDA% "-DCMAKE_CUDA_COMPILER:FILEPATH=!CUDA_NVCC!"
    )
) else (
    if defined QT_PREFIX (
        "%CMAKE_EXE%" -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTEM_ENABLE_CUDA=%ENABLE_CUDA% -DCMAKE_PREFIX_PATH="%QT_PREFIX%"
    ) else (
        "%CMAKE_EXE%" -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTEM_ENABLE_CUDA=%ENABLE_CUDA%
    )
)
if not "%ERRORLEVEL%"=="0" goto configure_failed

echo.
echo Compiling InverTEM...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release --parallel %PARALLEL_JOBS%
rem Negative exit codes (e.g. -1 from a failed link) are not caught by "if errorlevel 1".
if not "%ERRORLEVEL%"=="0" goto build_failed

rem Remove obsolete pre-InverTEM executables so File Explorer cannot launch
rem a stale build with the old product name or dialog text.
if exist "%BUILD_DIR%\pytem-inversion-gui.exe" del /q "%BUILD_DIR%\pytem-inversion-gui.exe" >nul 2>nul
if exist "%BUILD_DIR%\Release\pytem-inversion-gui.exe" del /q "%BUILD_DIR%\Release\pytem-inversion-gui.exe" >nul 2>nul

if exist "%BUILD_DIR%\InverTEM.exe" set "APP_EXE=%CD%\%BUILD_DIR%\InverTEM.exe"
if exist "%BUILD_DIR%\Release\InverTEM.exe" set "APP_EXE=%CD%\%BUILD_DIR%\Release\InverTEM.exe"
if not defined APP_EXE goto missing_executable

rem If g4.png is beside this batch file, CMake copies it beside InverTEM.exe.
rem Keep this fallback for existing build trees configured before the icon was added.
for %%I in ("!APP_EXE!") do set "APP_DIR=%%~dpI"
if exist "g4.png" copy /y "g4.png" "!APP_DIR!g4.png" >nul

rem Copy the matching MSVC Qt runtime beside the executable. This prevents
rem Windows from loading incompatible Qt DLLs from an MSYS2/MinGW PATH entry.
if defined QT_PREFIX (
    if exist "!QT_PREFIX!\bin\windeployqt.exe" (
        echo.
        echo Deploying the matching Qt runtime...
        "!QT_PREFIX!\bin\windeployqt.exe" --release --no-translations "!APP_EXE!"
        if errorlevel 1 goto deploy_failed
    ) else (
        echo WARNING: windeployqt.exe was not found; the executable may not start outside a Qt command prompt.
    )
)

rem Optional: one portable InverTEM_portable.exe (needs Enigma Virtual Box).
if exist "%~dp0Package_InverTEM.ps1" (
    echo.
    echo Packing a portable single-file executable...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Package_InverTEM.ps1" -AppExe "!APP_EXE!"
    if errorlevel 1 echo WARNING: the portable executable could not be made; the normal build is unaffected.
)

echo.
echo Build completed successfully.
echo Executable: "!APP_EXE!"
echo.
pause
exit /b 0

:find_msvc
rem nvcc on Windows uses Microsoft's 64-bit C/C++ compiler as its host compiler.
where cl.exe >nul 2>nul
if not errorlevel 1 goto msvc_ready

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do if not defined VS_INSTALL set "VS_INSTALL=%%I"
)

if not defined VS_INSTALL (
    for %%V in (18 2022) do for %%E in (BuildTools Community Professional Enterprise) do (
        if not defined VS_INSTALL if exist "%ProgramFiles%\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VS_INSTALL=%ProgramFiles%\Microsoft Visual Studio\%%V\%%E"
    )
)

if defined VS_INSTALL if exist "!VS_INSTALL!\VC\Auxiliary\Build\vcvars64.bat" (
    call "!VS_INSTALL!\VC\Auxiliary\Build\vcvars64.bat" >nul
)

where cl.exe >nul 2>nul
if errorlevel 1 exit /b 1

:msvc_ready
for /f "delims=" %%I in ('where cl.exe 2^>nul') do if not defined MSVC_CL set "MSVC_CL=%%~fI"
echo MSVC host compiler: !MSVC_CL!
exit /b 0

:find_cuda
rem Prefer CUDA_PATH, then PATH, then the newest toolkit in NVIDIA's default folder.
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_NVCC=%CUDA_PATH%\bin\nvcc.exe"
if defined CUDA_NVCC exit /b 0

for /f "delims=" %%I in ('where nvcc.exe 2^>nul') do if not defined CUDA_NVCC set "CUDA_NVCC=%%~fI"
if defined CUDA_NVCC exit /b 0

for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
    if not defined CUDA_NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D\bin\nvcc.exe" set "CUDA_NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D\bin\nvcc.exe"
)
exit /b 0

:find_qt_msvc
rem Prefer QTDIR when it names an MSVC kit, then scan Qt's standard folder.
if defined QTDIR if exist "%QTDIR%\lib\cmake\Qt6\Qt6Config.cmake" (
    echo %QTDIR% | findstr /I /C:"msvc" >nul
    if not errorlevel 1 set "QT_PREFIX=%QTDIR%"
)
if defined QT_PREFIX exit /b 0

for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Qt\6.*" 2^>nul') do (
    if not defined QT_PREFIX if exist "C:\Qt\%%D\msvc2022_64\lib\cmake\Qt6\Qt6Config.cmake" set "QT_PREFIX=C:\Qt\%%D\msvc2022_64"
)
exit /b 0

:location_failed
echo ERROR: Could not open the folder containing this batch file.
goto failed

:missing_project
echo ERROR: Place Build_InverTEM.bat beside CMakeLists.txt in the cpp_gui folder.
goto failed

:missing_cmake
echo ERROR: CMake was not found at "%CMAKE_EXE%".
goto failed

:missing_msvc
echo.
echo ERROR: CUDA was found, but Microsoft cl.exe was not found.
echo Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload.
echo Include MSVC v143 x64/x86 tools and a Windows 10 or Windows 11 SDK.
echo Then double-click this same batch file again.
goto failed

:configure_failed
echo.
echo CMake configuration failed.
echo Confirm that Ninja, a C++ compiler, and Qt 5.15 or Qt 6 Widgets are installed.
echo CUDA/MSVC requires Qt's MSVC 2022 64-bit kit, not the MSYS2/MinGW Qt kit.
echo Install that component with the Qt Maintenance Tool, then set QT_PREFIX near the top of Build_InverTEM.bat if it is not under C:\Qt.
echo If an installed CUDA toolkit causes configuration problems, set ENABLE_CUDA=OFF.
goto failed

:build_failed
echo.
echo Compilation failed. Review the messages above.
echo If the error is "cannot open file 'InverTEM.exe'" (LNK1104), close InverTEM and build again.
goto failed

:missing_executable
echo.
echo Compilation completed, but InverTEM.exe was not found.
goto failed

:deploy_failed
echo.
echo Compilation completed, but the matching Qt runtime could not be deployed.
echo Do not launch the executable until this step succeeds; Windows may otherwise load the old MinGW Qt DLLs.

:failed
echo.
pause
exit /b 1
