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
  with Python-matched alpha search, RMS-increase iteration advancement, and a
  log-alpha parabolic backtrack to the RMS=1 target
- Parallel step-response evaluation using standard C++ threads
- Optimized CPU recursion that precomputes layer conductivity terms, system
  filters, Hankel lambdas, and offset-loop Bessel factors outside the innermost
  frequency/layer loops
- Optional native NVIDIA CUDA acceleration for circular DLF forward responses,
  including the expensive alpha-trial calculations
- CUDA kernels precompute conductivity and offset-loop Hankel/Bessel geometry
  once per model and use a register-sized reflection recursion without a
  per-thread 128-layer temporary array
- Bounded parallel inversion of multiple USF soundings
- Shared first-iteration Jacobian cache in memory and on disk

## USF workflow

Open one or many TEMcompany `.usf` files in one selection. The GUI can invert
LM, HM, or LM + HM jointly for every sounding. Soundings run concurrently,
while each sounding's LM and HM share one layered-earth model. Each moment
retains its own gates, waveform matrix, and analog filter chain. Previous/next
buttons browse the imported soundings and their recovered models. A batch
progress bar is accompanied by elapsed time and an ETA that starts after the
first reported inversion iteration and refreshes at most once every ten
seconds. Internal work-unit counts are not shown. The native reader:

1. groups non-noise sweeps by channel and sorts moments by frequency;
2. labels the moments LM and HM;
3. stacks voltage per gate and calculates the standard error of the mean;
4. reads gate centers/open/close times, `TX_RAMP`, and analog filter stages;
5. initializes the geometry from `LOOP_SIZE` and `COIL_LOCATION`.

The reader also accepts WalkTEM Importer files that provide compact
`TIME, VOLTAGE, QUALITY` tables rather than explicit error/open/close columns.
Their quality flags control the initial gate selection. If gate edges are not
stored, logarithmic edges are derived from adjacent gate centres; if the file
declares an instantaneous ramp without a `TX_RAMP` array, the forward operator
uses an ideal turn-off waveform. WGS84 UTM coordinates declared with
EPSG:32601–32660 or EPSG:32701–32760 are converted to longitude/latitude for
background-map alignment.

Imported transmitter-loop and receiver-coil dimensions are displayed under
`Geometry:`. Gates with non-positive stacked voltage or SNR below 3 are
disabled initially. Right-clicking a point in the data plot toggles that gate.
LM data are blue and HM data are red, with vertical error bars taken from the
USF standard error for each gate. Time-axis tick labels use consistent
scientific notation. Before inversion the plot uses markers only;
after inversion the fitted response is added as a solid line. Disabled points
remain visible in a faded colour but do not add legend entries. Clicking a sounding on the map toggles
the entire sounding in or out of the next batch while preserving its individual
gate choices.

The starting model defaults to 15 layers at a fixed 100 Ohm m. The user chooses
the number of layers and the depth to the base of layer N-1; layer N is the
half-space. The first finite layer is exactly 1 m thick, and a single geometric
ratio is solved so the remaining thicknesses increase logarithmically and end
exactly at the requested depth. Resistivity bounds remain internal safeguards,
not GUI controls.
The `Parallel soundings` control caps concurrent jobs at the available hardware
thread count; CPU threads are divided among active inversions to avoid nested
oversubscription.

`Use NVIDIA GPU (CUDA)` is checked by default when the application was built
with CUDA and a compatible GPU is detected. Uncheck it for CPU-only execution.
The control is disabled when CUDA is unavailable or the Euler transform is
selected; analytical Jacobian/sensitivity assembly currently remains on CPU.

## Interface layout

- A resizable information sidebar to the left of the data plot contains the
  file-open button, previous/next sounding navigation, sounding and moment
  selectors, imported geometry, and the waveform/filter summary. A modal
  progress dialog reports multi-file import progress.
- The information sidebar, data plot, and log-resistivity plot share the
  upper-left workspace. The
  resizable sounding map occupies a wider dedicated full-height right column.
- The recovered-model depth axis begins at exactly 0 m and increases downward.
- Inversion controls and the run log are grouped in vertical columns below the
  two left plots.
- The window opens centered and sized to the available desktop rather than a
  fixed resolution.

Iteration and alpha-trial counts use fixed native solver defaults and are not
shown as editable menu controls.

The map uses equal horizontal and vertical coordinate scaling and displays an
automatically sized metre/kilometre scale bar. Its uncluttered axes are labelled
Easting and Northing without numeric tick labels, and its point legend is drawn
on an opaque white background. Left-clicking a
sounding navigates directly to its data and recovered model; right-clicking
also navigates to it, excludes it from the next batch, and greys out all of its
gates; right-clicking again restores its previous gate selection. Radio buttons at the bottom of the
map column select no background, OpenStreetMap, or satellite imagery. Selecting
an online map loads the visible 5 × 5 tile block with a dedicated progress bar
and a persistent HTTP cache. The tile mosaic is sized beyond the data-driven
viewport and clipped so the background fills the complete map plot while
remaining aligned with the soundings. Provider attribution is not displayed.
Dynamic content reserves its layout space, and the top-level layout is capped
to the available desktop, so map loading and right-click inclusion/exclusion do
not resize the main window beyond the screen or redistribute the plot splitters.

At batch start, a sounding is skipped when any selected moment has fewer than
three enabled gates. One warning lists every skipped sounding while valid
soundings continue into the inversion.

The **Kill inversion** button cooperatively stops active calculations and does
not start further queued soundings. Models that finished before the stop are
kept for navigation and export; incomplete soundings are discarded.

After every batch, a tab-separated `pytem_inversion_timing_*.txt` report is
written beside the first included USF file. It records preparation, wall time,
initial forward calculation, Jacobians, linear solves, alpha-trial forwards,
final sensitivity, and unclassified overhead for every sounding. This is
intended for direct performance comparison with the Python inversion.

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
- CMake 3.24+
- Qt 6 Widgets, or Qt 5.15 Widgets
- Qt Network from the same Qt installation (for optional online maps)
- Optional: NVIDIA CUDA Toolkit and compatible GPU for DLF forward acceleration
- For CUDA on Windows: Visual Studio 2022/2026 or Build Tools with the
  **Desktop development with C++** workload, plus a matching Qt MSVC kit

No Python, NumPy, or pyTEM installation is required by the application.

CMake detects CUDA automatically. Use `-DPYTEM_ENABLE_CUDA=OFF` to force a
CPU-only build. Without a CUDA toolkit the same source builds normally and the
GPU checkbox is disabled at runtime. CUDA compilation passes
`/Zc:preprocessor` to the MSVC host compiler, as required by the CCCL headers
shipped with recent CUDA 13.x toolkits. Native C++ compilation also passes
`/Zc:__cplusplus`, as required by Qt's MSVC headers.

An MSVC/CUDA build must use a Qt kit compiled for MSVC. The Qt libraries below
`C:\msys64\ucrt64` are MinGW libraries and are not binary-compatible with
MSVC. CMake rejects that combination during configuration with a direct
diagnostic rather than allowing a later header or linker failure.

## Build on Windows

With Qt's MSVC kit installed:

```powershell
cmake -S cpp_gui -B cpp_gui/build -DCMAKE_PREFIX_PATH="C:\Qt\6.8.0\msvc2022_64"
cmake --build cpp_gui/build --config Release --parallel
& cpp_gui\build\Release\pytem-inversion-gui.exe
```

With MinGW, point `CMAKE_PREFIX_PATH` at the matching Qt MinGW directory and
use that compiler's CMake generator.

Place the included `Build_pyTEM.bat` beside `CMakeLists.txt` in the `cpp_gui`
folder and double-click it. The single batch file configures or refreshes the
`build-ninja` directory, compiles the Release target with four parallel jobs,
and stops without launching the application. It locates `nvcc.exe` from
`CUDA_PATH`, `PATH`, or NVIDIA's standard Toolkit directory and supplies the
full path to CMake. When CUDA is enabled, it uses Visual Studio Installer's
`vswhere.exe` to locate the Microsoft C++ tools and calls `vcvars64.bat`, which
places `cl.exe` and the required SDK environment on the build path. A previous
CPU-only, incomplete CUDA, or MinGW-Qt cache is refreshed automatically. The
batch file automatically selects the newest
`C:\Qt\6.*\msvc2022_64` kit when present. If the Microsoft
C++ workload is missing, the batch stops with installation instructions. If Qt
is not discoverable automatically, set `QT_PREFIX` near the top of the batch
file to the matching MSVC Qt kit directory, such as
`C:\Qt\6.8.0\msvc2022_64`. `ENABLE_CUDA=ON` enables CUDA detection; change it
to `OFF` if a CPU-only build is required. After compilation, the batch runs
the matching MSVC `windeployqt` and copies the required Qt DLLs and platform
plugins beside the executable. This prevents Windows from loading incompatible
Qt DLLs from an MSYS2/MinGW `PATH` entry when the application is launched from
File Explorer.

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

The native tests check CPU/GPU-fallback DLF and Euler transforms against fixed pyTEM reference
values, compare the analytical Jacobian with finite differences through the
filter and response-matrix chain, exercise the Python-style parabolic RMS=1
target fit and joint LM+HM inversion, verify L1
regularization and memory/disk Jacobian reuse, and test the matrix builder. The
complete filtered waveform/gate
chain was cross-checked on `L008_S001_2026_0915_084148.usf`: median relative
difference from the Kenbec Python matrix path was below 0.00001%, with a
maximum below 0.09% over the LM gates.
