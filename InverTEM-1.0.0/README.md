# InverTEM 1.0.0

A self-contained C++17/Qt desktop application for 1-D circular-loop TEM
forward modelling and inversion. The executable does not launch Python and has
no Python runtime dependency.

## Numerical core

- Wait layered-earth TE reflection-coefficient recursion
- Euler-11 inverse Laplace (default) or Key 81-point Fourier DLF transform
  with a shared Key 101-point Hankel filter
- Central and radial-offset circular transmitter/receiver geometries
- First- and second-order low/high-pass stages applied in the frequency domain
- Kenbec-style quintic B-spline matrix operator for transmitter waveform
  convolution and finite receiver-gate averaging
- Analytical adjoint (default) or finite-difference log-resistivity Jacobian;
  both propagate through the band-pass stages and waveform/gate matrix
- Optional guarded Broyden rank-one Jacobian updates between periodically
  refreshed full Jacobians
- Selectable L2-smooth or L1-blocky regularized Gauss-Newton inversion
  with a guarded adaptive or reproducible fixed alpha search, RMS-increase
  iteration advancement, and a single-evaluation local log-alpha parabolic
  fit to the RMS=1 target. Adaptive search is independent per sounding; an
  invalid or unstable trial retains the best finite model and falls back to
  the fixed sweep for that sounding.
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

Create an InverTEM project (`.ivtp`), load an existing project, or begin in an
unsaved project and save it later. A project stores its USF references, gate
and sounding selections, inversion settings, map/axis preferences, and all
saved inversion models. Relative and absolute USF paths are retained so a
project remains usable when its folder is moved together with the data.
The parsed sounding data is also embedded in the project, so a saved project
reopens even when the USF files are unavailable; projects saved by older
versions still read their USF files.

The first accepted USF locks the project's transmitter-loop and receiver-coil
geometry. Later imports are additive and only matching geometry is accepted;
files with different `LOOP_SIZE` or `COIL_LOCATION` values are skipped with a
warning that shows the expected and encountered dimensions. Survey position,
elevation, and coordinate reference system may vary between soundings and are
not part of the system-geometry lock. A new project starts with an unlocked
geometry.

Open one or many `.usf` files in one selection. The GUI can invert
LM, HM, or LM + HM jointly for every sounding. Soundings run concurrently,
while each sounding's LM and HM share one layered-earth model. Each moment
retains its own gates, waveform matrix, and analog filter chain. Previous/next
buttons browse the imported soundings and their recovered models. A batch
progress bar is accompanied by elapsed time and an ETA that starts after the
first reported inversion iteration and refreshes at most once every ten
seconds. The adjacent status field shows only the completed-sounding count,
for example `37/288 completed`; per-sounding RMS, DOI, and solver messages
remain in the log and result exports. Internal work-unit counts are not shown.
The native reader:

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
Holding the right button and dragging a rectangle disables all enabled points
inside it; if the rectangle contains only disabled points, those points are
restored together.
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
The `Batch workers` control caps the shared worker pool at the available
hardware thread count. InverTEM displays the logical CPU core/thread count
reported by the operating system and prevents the worker setting from
exceeding it. Soundings remain mathematically independent, but active
inversions advance through synchronized forward, Jacobian, and alpha-trial
stages. The project has one locked system geometry. Its compatible starting
responses and analytical Jacobians are evaluated on one union of the selected
gates per moment (LM and HM have separate waveform/filter chains), then the
exact rows are subselected for each independent sounding. Different gate masks
therefore no longer repeat the expensive waveform step-time grid. Existing
on-disk first-Jacobian entries are detected before this calculation; when all
required entries are cached, only the shared starting forward response is
evaluated. Other candidates are distributed across the shared pool without
nested thread oversubscription. OpenMP is used automatically when available,
with a standard C++ thread fallback.

This is a block-diagonal scheduling optimization, not a laterally constrained
inversion: no sounding contributes data or regularization to another sounding.
Each sounding retains its own RMS history, alpha decisions, stopping condition,
gate selection, and final model. Finished soundings leave the active batch and
are retained immediately, including when a later kill request stops the
remaining work. If the synchronized path encounters an unsupported dataset, it
falls back to the original independent job runner.

The small dense normal equations use partial-pivot LU elimination with
back-substitution. Intel MKL PARDISO is intentionally not required: PARDISO is
well suited to a single large sparse laterally constrained system, whereas the
independent InverTEM problems are many small dense systems for which its setup
cost would dominate.

The model plot provides an automatic resistivity-axis mode plus editable
minimum and maximum limits. Logarithmic resistivity ticks use sensible 1/2/5
intervals instead of arbitrary evenly spaced values.

`Use NVIDIA GPU (CUDA)` is checked by default when the application was built
with CUDA and a compatible GPU is detected. Uncheck it for CPU-only execution.
The control is disabled when CUDA is unavailable or the Euler transform is
used; analytical Jacobian/sensitivity assembly currently remains on CPU.

## Interface layout

- A resizable information sidebar to the left of the data plot contains the
  New/Load/Save project controls, the additive file-import button,
  previous/next sounding navigation, sounding and moment
  selectors, imported geometry and filter/gate summary. A modal
  progress dialog reports multi-file import progress.
- The data-view selector can replace the single-sounding data/model pair with
  stacked LM and HM distance transects. The spin box beside the selector sets
  how many soundings each page contains (default 30, saved in the project);
  previous/next controls page by that count. Pages support the same right-click and
  right-drag gate editing, and highlights its visible stations on the map.
- The information sidebar, data plot, and log-resistivity plot share the
  upper-left workspace. The
  resizable sounding map occupies a wider dedicated full-height right column.
- The recovered-model depth axis begins at exactly 0 m and increases downward.
- Inversion controls and the run log are grouped in vertical columns below the
  two left plots.
- The window opens centered and sized to the available desktop while remaining
  freely resizable.

Iteration and alpha-trial counts use fixed native solver defaults and are not
shown as editable menu controls.

`Adaptive alpha search` is enabled by default and can be turned off before a
batch to reproduce the original fixed descending sweep. The setting is locked
while a batch is running and is stored in the project file.

## Solver settings file

The inversion method, transform, Jacobian method and Jacobian update have no GUI controls. They
are read at start-up from `invertem_solver.txt` beside the executable, which is
written with the defaults on first run:

```
method = pytem_joint       # pytem_joint | adaptive (InverTEM adaptive Gauss-Newton)
transform = euler          # euler | dlf (DLF enables the CUDA option)
jacobian = analytical      # analytical | finite_difference (not used by pyTEM joint)
jacobian_update = full     # full | broyden (not used by pyTEM joint)
broyden_refresh = 3        # iterations between full Jacobians with broyden (2-10)
```

The `sci_*` keys set the Fast SCI constraint strengths (see below).
The timing report records the values used. `jacobian_update = broyden` uses a
guarded Broyden rank-one update instead of full recalculation every iteration. The refresh interval limits how many accepted
updates may occur before a full analytical/finite-difference Jacobian is
rebuilt; unstable rank-one corrections trigger an immediate rebuild.

Once an alpha trial brackets RMS 1, the solver fits only the local two or
three points and performs one final target forward calculation. If the first
alpha already undershoots, it performs one interpolated target calculation
instead of searching repeatedly with stronger alpha values.

The map uses equal horizontal and vertical coordinate scaling and displays an
automatically sized metre/kilometre scale bar. Its uncluttered axes are labelled
Easting and Northing without numeric tick labels, and its point legend is drawn
on an opaque white background. A single loaded sounding receives a compact
local extent so its point and background tiles remain visible. Left-clicking a
sounding navigates directly to its data and recovered model; right-clicking
also navigates to it, excludes it from the next batch, and greys out all of its
gates; right-clicking again restores its previous gate selection. Radio buttons at the bottom of the
map column select no background, OpenStreetMap, or satellite imagery. Selecting
an online map loads the visible 7 × 7 tile block with a dedicated progress bar
and a persistent HTTP cache. The tile mosaic is sized beyond the data-driven
viewport and clipped so the background fills the complete map plot while
remaining aligned with the soundings. Provider attribution is not displayed.
As each sounding completes during a running batch, its marker is immediately
filled and coloured by final RMS on a fixed 0.00–5.00 green–yellow–red scale;
its colour then remains fixed. Uninverted and currently running soundings
remain hollow. A point-size menu below the map provides small, medium, large,
and extra-large markers and saves the selection in the project. The complete
RMS colour scale and distance scale bar are each drawn on an opaque white,
framed panel so they remain readable over satellite imagery. Dynamic
content reserves its layout space, so map loading and right-click
inclusion/exclusion do not redistribute the plot splitters.

At batch start, a sounding is skipped when any selected moment has fewer than
three enabled gates. One warning lists every skipped sounding while valid
soundings continue into the inversion.

The **Kill inversion** button cooperatively stops active calculations and does
not start further queued soundings. Models that finished before the stop are
kept for navigation and export; incomplete soundings are discarded.

Completed models are retained per sounding and configuration. Running L1 and
then L2, or repeating an inversion with another layer count or maximum depth,
adds another labelled curve to the model plot instead of replacing the earlier
model. Repeating the same configuration updates that saved entry. **Clear saved
results** releases all retained inversion models from memory without deleting
the persistent first-Jacobian cache.

When final sensitivity is enabled, DOI is the interpolated depth containing
95% of the cumulative finite-layer information in the noise-weighted
Jacobian. The half-space is excluded because it has no finite bottom. The
model curve is solid above DOI and dashed below it; no DOI boundary line or
grey overlay is drawn. DOI is also recorded in JSON/CSV exports.
The dedicated **Export modelled data (.xyz)** button writes the selected saved
configuration for all completed soundings using the Aarhus Workbench XYZ
layout: coordinates and elevation, `RHO`, `RHO_STD`, `DEP_BOT`,
`DEP_BOT_STD`, conservative/standard DOI, and final RMS (`RESDATA`). Unknown
standard deviations are written as `nan`.

After every batch, a tab-separated `invertem_timing_*.txt` report is
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

Elapsed time refreshes every second while ETA is deliberately recalculated at
most every ten seconds. Displayed floating-point status values use two decimal
places. An RMS up to 1.004 is accepted as converged because it displays as
1.00.

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

No Python or NumPy installation is required by the application.

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
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\Qt\6.8.0\msvc2022_64"
cmake --build build --config Release --parallel
& .\build\Release\InverTEM.exe
```

With MinGW, point `CMAKE_PREFIX_PATH` at the matching Qt MinGW directory and
use that compiler's CMake generator.

Double-click the included `Build_InverTEM.bat` beside `CMakeLists.txt`. The
single batch file configures or refreshes the
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
File Explorer. After a successful build it also removes any obsolete
`pytem-inversion-gui.exe`; launch `InverTEM.exe`.

## Build on Linux/macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/InverTEM
```

Or use the included convenience Makefile:

```bash
make          # configure and build
make test     # build and run native tests
make run      # build and launch the GUI
```

## Tests

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The native tests check CPU/GPU-fallback DLF and Euler transforms against fixed Python reference
values, compare the analytical Jacobian with finite differences through the
filter and response-matrix chain, exercise the Python-style parabolic RMS=1
target fit and joint LM+HM inversion, verify L1
regularization and memory/disk Jacobian reuse, and test the matrix builder. The
complete filtered waveform/gate
chain was cross-checked on `L008_S001_2026_0915_084148.usf`: median relative
difference from the Kenbec Python matrix path was below 0.00001%, with a
maximum below 0.09% over the LM gates.

## Fast SCI

Choosing **SCI** as the regularisation (log data space; a linear-data-space
variant exists in the code but is hidden) inverts
every included sounding at once, after the Lupus scheme (TEMcompany):

- one joint objective: data misfit in log data space (ln d, continued linearly
  below 1/1000 of the observed value so sign-changing responses stay finite)
  or linear data space (d, as Lupus), vertical constraints
  between adjacent layers and spatial constraints between neighbouring
  soundings (Delaunay triangulation of the sounding positions);
- lateral constraints compare each layer with the neighbour's resistivity at
  the same elevation (from the soundings' ground elevations, interpolated
  between the neighbour's layer mid-depths; nothing is constrained above the
  neighbour's surface). `sci_constraints = depth` compares equal depths below
  the surface instead; on flat ground both are identical;
- a lateral constraint loosens with distance as
  `ln(sci_lateral_factor) * (distance / sci_reference_distance)^sci_distance_power`;
- each sounding starts from its best-fitting homogeneous half-space, chosen
  from 33 resistivities between 1 Ohm m and 10 kOhm m (8 per decade); these
  half-space responses depend only on the system, so they are computed once
  and shared by all soundings;
- each iteration uses one Jacobian per sounding and a Marquardt damping ladder
  (x3 per level, starting three levels below the last accepted one). The first
  step whose largest change in ln(resistivity) is within the step limit and
  that lowers the median per-sounding RMS (always measured in log space, so a
  few badly fitting soundings cannot dominate it) by at least 0.7% is accepted. The limit starts at 3,
  grows x1.2 per iteration and shrinks when progress stalls; the inversion ends
  when it reaches 1.1 without progress, when the median log-space RMS of the
  soundings still in the system reaches 1 (half fit, so the rest are not
  over-fitted while outliers lag), or after 30 iterations;
- a sounding whose own misfit would more than double under a batch step backs
  off towards its current model (up to four halvings, else it keeps its model
  for that iteration);
- a sounding still above RMS 1 that improved by less than 1% over its last 3
  iterations is abandoned: its model is frozen and it leaves the data,
  constraint and median terms. Its log message says when;
- reported RMS values are log-space RMS, as for the other methods;
- the coupled system is solved by conjugate gradients preconditioned with a
  Cholesky factor per sounding.

`invertem_solver.txt` is read at start-up and again before every batch, so
edits apply to the next run without restarting. Missing settings (including
the `sci_*` ones in older files) are appended with their current values.
Defaults: vertical factor 3.0, lateral factor 1.5 at 100 m, distance power 0.5,
30 iterations; larger factors mean looser constraints.

## Portable single-file executable

After deployment, `Build_InverTEM.bat` runs `Package_InverTEM.ps1`, which uses
the free Enigma Virtual Box (https://enigmaprotector.com/en/downloads.html) to
pack `InverTEM.exe`, the Qt DLLs and plugins, `g4.png` and the Microsoft C++ /
OpenMP runtime into `InverTEM_portable.exe` beside the normal executable. That
one file can be shared and run on another Windows PC without installing
anything; its settings file, projects and reports are written next to it as
usual. The OpenGL/Direct3D fallback DLLs are left out (InverTEM draws with the
raster engine). If Enigma Virtual Box is not installed (or `ENIGMA_VB_CONSOLE`
does not point to `enigmavbconsole.exe`), the step is skipped with a note.

The same step also writes `InverTEM_portable.zip`: the exe with the same runtime
files in one plain folder. Share this if the single-file exe will not start on
another PC. Packed exes are often blocked by antivirus or SmartScreen (on the
first run choose *More info → Run anyway*, or right-click → Properties →
*Unblock*), and the unzipped folder avoids that. Both need 64-bit Windows 10
(1809 or later) or Windows 11, which Qt 6 requires.

## Automatic gate selection

Imported soundings get their gates chosen automatically, per moment:

1. accepted quality flag, positive voltage and SNR >= 3;
2. nothing after the first negative voltage (a step-off decay does not change sign);
3. gates that break a smooth, falling decay are culled one at a time, worst
   first: each gate is compared with a quadratic in log(time)-log(voltage)
   through its 6 nearest kept gates, with a tolerance of max(0.2, 3 x relative
   error) in ln V, and a voltage that rises over the previous gate is culled;
4. from the first culled gate after which most gates are culled, the late-time
   tail is cut; a moment where most gates fail is dropped entirely.

**Auto-filter gates** (next to the export buttons) re-applies this to every
sounding, replacing manual gate edits. A moment left without gates is simply
left out of the inversion (e.g. LM only); a sounding is skipped only when a
fitted moment keeps fewer than three gates.

## Editing gates and soundings

- Data and transect plots: **right-click** a gate to toggle it, **right-drag** a
  box to remove every gate in it, **Shift + right-drag** to restore them.
  **Ctrl+Z** undoes the last gate, auto-filter or map inclusion edit.
- After a gate edit the model stays on screen, marked "(gates changed)", with
  the predicted response dashed, until the sounding is inverted again.
- Map: left-click selects a sounding, right-click excludes or re-includes it.
  Excluding does not change its gates; they are shown greyed while excluded
  and cannot be edited until it is included again.
