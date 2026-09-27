# pyTEM native C++ inversion GUI

A self-contained C++17/Qt desktop application for 1-D circular-loop TEM
forward modelling and inversion. The executable does not launch Python and has
no Python runtime dependency.

## Numerical core

- Wait layered-earth TE reflection-coefficient recursion
- Selectable Key 81-point Fourier DLF or Euler-11 inverse Laplace transform
  with a shared Key 101-point Hankel filter
- Central and radial-offset circular transmitter/receiver geometries
- First- and second-order low/high-pass stages applied in the frequency domain
- Kenbec-style quintic B-spline matrix operator for transmitter waveform
  convolution and finite receiver-gate averaging
- Selectable analytical adjoint or finite-difference log-resistivity Jacobian;
  both propagate through the band-pass stages and waveform/gate matrix
- Selectable L2-smooth or L1-blocky (IRLS) regularized Gauss-Newton inversion
  with alpha search, bounds, and backtracking
- Parallel step-response evaluation using standard C++ threads
- Bounded parallel inversion of multiple USF soundings
- Shared first-iteration Jacobian cache in memory and on disk

## USF workflow

Open one or many TEMcompany `.usf` files in one selection. The GUI can invert
LM, HM, or LM + HM jointly for every sounding. Soundings run concurrently,
while each sounding's LM and HM share one layered-earth model. Each moment
retains its own gates, waveform matrix, and analog filter chain. Previous/next
buttons browse the imported soundings and their recovered models. The native
reader:

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

The starting model defaults to 15 layers. Its 14 finite-layer thicknesses are
logarithmically spaced from 2 to 25 m; the final layer is a half-space. The
layer-count control regenerates that thickness sequence for any selected count.
The `Parallel soundings` control caps concurrent jobs at the available hardware
thread count; CPU threads are divided among active inversions to avoid nested
oversubscription.

Each selected moment's matrix operator is assembled once when Run is pressed.
Its waveform and finite gate widths are combined into a fixed matrix `M` on a
300-point log-time step-response grid. Every model evaluation then uses one
matrix multiplication per moment:

```text
predicted_gates = M * filtered_step_response
```

## First-Jacobian reuse

When `Reuse/cache first Jacobian` is enabled, parallel inversions with an
identical starting model and forward configuration calculate the first
Jacobian only once. Other jobs wait for and reuse that matrix. The matrix is
also stored below the operating system's application cache directory for
future runs.

The cache signature includes layer geometry and starting resistivities, gate
times, waveform matrices, filter stages, loop geometry, transform, and
Jacobian method. Any difference forces a new calculation; observed voltages
and error bars are deliberately excluded because they do not affect the
Jacobian. Later iterations still calculate their own Jacobians because their
models diverge independently.

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

Or, from `cpp_gui`, use the included convenience Makefile:

```bash
make          # configure and build
make test     # build and run native tests
make run      # build and launch the GUI
```

## Tests

```bash
cmake -S cpp_gui -B cpp_gui/build -DBUILD_TESTING=ON
cmake --build cpp_gui/build --parallel
ctest --test-dir cpp_gui/build --output-on-failure
```

The native tests check both C++ transforms against fixed pyTEM reference
values, compare the analytical Jacobian with finite differences through the
filter and response-matrix chain, exercise joint LM+HM inversion, verify L1
regularization and memory/disk Jacobian reuse, and test the matrix builder. The
complete filtered waveform/gate
chain was cross-checked on `L008_S001_2026_0915_084148.usf`: median relative
difference from the Kenbec Python matrix path was below 0.00001%, with a
maximum below 0.09% over the LM gates.
