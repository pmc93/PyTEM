@echo off
setlocal
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "CTEST=C:\Program Files\CMake\bin\ctest.exe"
set "TOOLCHAIN=C:\msys64\ucrt64"
set "PATH=%TOOLCHAIN%\bin;%PATH%"
set "SOURCE=%~dp0."
set "BUILD=%~dp0build-ninja"

for %%F in ("%CMAKE%" "%CTEST%" "%TOOLCHAIN%\bin\g++.exe" "%TOOLCHAIN%\bin\ninja.exe" "%TOOLCHAIN%\lib\cmake\Qt6\Qt6Config.cmake") do (
    if not exist "%%~F" (
        echo Missing dependency: %%~F
        goto :failed
    )
)

echo Configuring Release build...
"%CMAKE%" -S "%SOURCE%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%TOOLCHAIN%" -DCMAKE_CXX_COMPILER="%TOOLCHAIN%\bin\g++.exe" -DCMAKE_MAKE_PROGRAM="%TOOLCHAIN%\bin\ninja.exe" -DBUILD_TESTING=ON
if errorlevel 1 goto :failed

echo Building...
"%CMAKE%" --build "%BUILD%" --parallel 4
if errorlevel 1 goto :failed

echo Running tests...
"%CTEST%" --test-dir "%BUILD%" --output-on-failure
if errorlevel 1 goto :failed

echo.
echo Build and tests succeeded.
echo Executable: "%BUILD%\pytem-inversion-gui.exe"
if /I not "%~1"=="--no-pause" pause
exit /b 0

:failed
echo.
echo Build or tests failed. See the error above.
if /I not "%~1"=="--no-pause" pause
exit /b 1