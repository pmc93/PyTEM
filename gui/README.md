# pyTEM native C++ inversion GUI

A self-contained C++17/Qt desktop application for 1-D circular-loop TEM
forward modelling and inversion. The executable does not launch Python and has
no Python runtime dependency.

## Numerical core

- Wait layered-earth TE reflection-coefficient recursion
- Key 101-point Hankel and 81-point Fourier digital linear filters
- Central and radial-offset circular transmitter/receiver geometries
- First- and second-order low/high-pass stages applied in the frequency domain
- Kenbec-style quintic B-spline matrix operator for transmitter waveform
  convolution and finite receiver-gate averaging
- Fixed-thickness, log-resistivity finite-difference Jacobian
- Smooth regularized Gauss-Newton inversion with alpha search, bounds, and
  backtracking
- Parallel step-response evaluation using standard C++ threads

## USF workflow

Open a TEMcompany `.usf` file directly. The native reader:

1. groups non-noise sweeps by channel and sorts moments by frequency;
2. labels the moments LM and HM;
3. stacks voltage per gate and calculates the standard error of the mean;
4. reads gate centers/open/close times, `TX_RAMP`, and analog filter stages;
5. initializes the geometry from `LOOP_SIZE` and `COIL_LOCATION`.

The current app is circular-loop only. A rectangular USF loop is represented
by its equal-area circular radius `sqrt(loop_x * loop_y / pi)`, and this
approximation is shown explicitly in the interface. Gates with non-positive
stacked voltage or SNR below 3 are unchecked initially; the user can change
gate selection before inversion.

The matrix operator is assembled once when Run is pressed. The selected
moment's waveform and finite gate widths are combined into a fixed matrix `M`
on a 300-point log-time step-response grid. Every model evaluation then uses
one matrix multiplication:

```text
predicted_gates = M * filtered_step_response
```

## Dependencies

- C++17 compiler
- CMake 3.21+
- Qt 6 Widgets, or Qt 5.15 Widgets

No Python, NumPy, or pyTEM installation is required by the application.

## Build on Windows

With Qt's MSVC kit installed:

```powershell
cmake -S cpp_gui -B cpp_gui/build -DCMAKE_PREFIX_PATH="C:\Qt\6.8.0\msvc2022_64"
cmake --build cpp_gui/build --config Release --parallel
& cpp_gui\build\Release\pytem-inversion-gui.exe
```

With MinGW, point `CMAKE_PREFIX_PATH` at the matching Qt MinGW directory and
use that compiler's CMake generator.

## Build on Linux/macOS

```bash
cmake -S cpp_gui -B cpp_gui/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_gui/build --parallel
./cpp_gui/build/pytem-inversion-gui
```

## Tests

```bash
cmake -S cpp_gui -B cpp_gui/build -DBUILD_TESTING=ON
cmake --build cpp_gui/build --parallel
ctest --test-dir cpp_gui/build --output-on-failure
```

The native tests check the C++ DLF response against fixed pyTEM reference
values and exercise the matrix builder. The complete filtered waveform/gate
chain was cross-checked on `L008_S001_2026_0915_084148.usf`: median relative
difference from the Kenbec Python matrix path was below 0.00001%, with a
maximum below 0.09% over the LM gates.

