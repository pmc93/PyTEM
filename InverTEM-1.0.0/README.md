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

Aarhus Workbench XYZ data exports (for example tTEM `Proc_AVG_export.xyz`) are
read too. Each row is one moment; rows with the same DATE and TIME form one
sounding, named `Line<LINE_NO>_<n>` and shown as `file.xyz#Line1_2`. Dummy
values mark gates as rejected, `DBDT_STD` is taken as the relative error, and
dB/dt in V/Am^4 is multiplied by `TX_AREA` to give V/Am². The XYZ carries no
system description, so InverTEM takes it from a system file beside it: a `.gex`
of the same name (else the only `.gex` in the folder), or a raw TEMcompany
`stb2xyz` file of the survey (for example `2026_0826_081707_ChA.xyz`). From the
TEMcompany file it reads `TxLoop_XYLength`, the Tx/Rx positions and heights
(`TxLoop_XYZPos`, `RxCoil_XYZPos`), `LPFilter_RxCoil`/`LPFilter_RxInst`, the LM
and HM waveforms, and the processed gate open/close times shifted by
`LM_GateTimeShift`/`HM_GateTimeShift` (the Workbench gate centres already are).
If none is found, InverTEM asks for one; cancelling uses tTEM defaults: 2 × 4 m
loop, receiver 9 m behind, Tx/Rx 0.5 m above ground, 670 kHz receiver filter,
LM 200 µs on / 2.5 µs turn-off at 2110 Hz, HM 450 µs on / 4 µs turn-off at
660 Hz. The log reports which was used.

Raw TEMcompany `stb2xyz` data files (first line `TEMcompany - stb2xyz.exe ...`)
can be imported directly; their header is the system. Each row is one LM or HM
stack, and consecutive LM and HM rows form one sounding (`Sounding_<n>`, about
22,000 in a full tTEM day, unaveraged). dB/dt in V/m² is divided by `TxCurrent`
and multiplied by `LM_DataFactor`/`HM_DataFactor`; gate centres are
`*_CenterTime` shifted by `*_GateTimeShift`, `dbdtStd` is the relative error, and
longitude/latitude are also projected to UTM for the model export.

For XYZ imports (not USF) InverTEM also asks for a line file (`.lin`, lines of
`date time line lat lon ! Start|End`). Records outside every Start–End interval
— the turns between lines — are not imported, and the soundings take the line
numbers of the file. Cancelling imports all data. Workbench exports are usually
already cut to the lines; for raw TEMcompany data the Vechta line file drops
about 14,000 of 44,000 records.

The model XYZ export writes UTM coordinates: the data's own UTM coordinates when
it has them, otherwise longitude/latitude projected into the UTM zone of the
survey's mean longitude, with the matching EPSG code in the header.

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

## SCI fast and SCI adaptive

Choosing **SCI fast** as the regularisation (log data space; a linear-data-space
variant exists in the code but is hidden) inverts
every included sounding at once, after the Lupus scheme (TEMcompany):

- one joint objective: data misfit in log data space (ln d, continued linearly
  below 1/1000 of the observed value so sign-changing responses stay finite)
  or linear data space (d, as Lupus), vertical constraints
  between adjacent layers and spatial constraints between neighbouring
  soundings (Delaunay triangulation of the sounding positions);
- lateral constraints compare each layer with the neighbour's layers it
  overlaps at the same elevation (from the soundings' ground elevations),
  weighted by the overlap; nothing is constrained above the neighbour's
  surface. `sci_constraints = depth` compares equal depths below the surface
  instead; on flat ground both are identical. Elevation constraints need
  heights that agree between passes: when soundings on different passes
  within 3 m typically differ by more than 1 m (GPS height drift), depth
  constraints are used and the log says so;
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

**SCI adaptive** chooses the constraint strength by the discrepancy principle:
the log vertical and lateral factors are scaled 0.25, 0.35, 0.5, 0.7, 1, 1.4,
2, 2.8 and 4 times (strong to loose), each run starting from the previous
models and running to convergence (no stop at median RMS 1, which would leave
poorly fitting lines behind as stripes), and the first run whose total RMS
over all soundings reaches 1 is kept (else the loosest). The chosen factors
are in the model name and the log. It takes several SCI fast runs.

`invertem_solver.txt` is read at start-up and again before every batch, so
edits apply to the next run without restarting; a removed line returns to its
default. Missing settings (including the `sci_*` ones in older files) are
appended with their current values. Defaults: vertical factor 3.0, lateral
factor 1.5 at 100 m, distance power 0.5, 30 iterations; larger factors mean
looser constraints. The SCI keys take comma-separated lists
(`sci_lateral_factor = 1.5, 2.5, 4`): the batch then runs every combination in
turn and keeps each as its own model ("SCI fast v3 l2.5 100 m p0.5"), selectable
per sounding and exportable. `invertem_solver_test.txt` is an example that
cycles through 18 combinations. The log lists the settings of every run when
the batch starts.

## Ground elevations from a terrain model

GPS heights are coarse (tTEM stores whole metres) and can drift by several
metres between passes, which misplaces soundings in sections and upsets
elevation constraints. **Ground elevations from DEM...** replaces them with
terrain heights (bilinear) from:

- DEM files: GeoTIFF (classic TIFF, strips or tiles, uncompressed, LZW or
  Deflate, predictors 1-3; the Cloud-Optimized GeoTIFFs of the Lower Saxony
  DGM1 or the Danish DHM work as downloaded) or an `x y z` text grid. Several
  tiles can be selected at once. UTM tiles (EPSG 258xx/326xx) are sampled at
  the soundings' positions in that zone, geographic ones (4326/4258) at their
  longitude/latitude;
- Copernicus GLO-90 (90 m cells) from the Open-Meteo elevation API
  (https://open-meteo.com/en/docs/elevation-api, free for non-commercial use
  with attribution to Copernicus and Open-Meteo), for positions rounded to
  about 50 m, 100 per request.

Soundings outside the model keep their heights; the log reports how many
changed and by how much. The corrected heights are saved with the project and
used by SCI and the model export.

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

If the exe will not start on another PC, antivirus or SmartScreen may be blocking
it (on the first run choose *More info → Run anyway*, or right-click →
Properties → *Unblock*). It needs 64-bit Windows 10 (1809 or later) or Windows
11, which Qt 6 requires.

Every start writes `invertem_log.txt` next to the exe (or in `%TEMP%` if that
folder is read-only): system and screen details, the Qt plugin loading, each
start-up step, all Qt warnings and, on a crash, the exception code and the DLL
it happened in. Ask for this file when InverTEM fails on another PC.

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

## Processing tools (after EEMstudio)

Ideas taken from the EEM Team's EEMstudio QGIS plugin:

- **Average soundings...** averages XYZ data along each line in windows of time
  or distance. Every gate averages the soundings within its own width of the
  window centre; the width is interpolated in log time between three (time,
  width) points per moment, so late gates average more soundings. The mean is
  weighted by 1/STD², drops values beyond two standard deviations, and its STD
  adds the minimum STD in quadrature. The raw soundings stay behind the
  averaged decay in grey and can be restored from the same dialog until
  InverTEM is closed.
- **Noise model...** sets STD = √(base² + (N·(t/1 ms)^-½ / |dB/dt|)²), with the
  base a uniform STD or the data's own and N a dB/dt noise level [V/m²] per
  moment, drawn on the decay plot. N starts at the level estimated from the
  data's own STDs (median over all soundings of the STD beyond the uniform one,
  scaled to 1 A and 1 ms). The window stays open while browsing, and the
  original STDs can be reset. The inversion still uses at least the error floor.
- **Remove STD ≥ N...** removes every used gate with a relative STD of N % or more.
- **Selections**: Ctrl + right-drag on a decay or transect selects gates. Then
  Q removes and A restores them, N removes the negative ones, E removes from
  the first selected gate to the last gate and W from the first gate to the
  last selected one (Shift+E / Shift+W: for every sounding of the page), 1, 2
  and 5 add 10, 20 and 50 % to their STD, 0 restores it, Esc clears.
- The data can be shown as late-time apparent resistivity (drop-down under "Data view").
- B bookmarks the current sounding (black on the map, ★ in the list) and Ctrl+B
  goes to the next bookmark.
- Sounding groups: the "Show" drop-down under the Group box shows and steps through
  one group only (data, models and transect); "Colour: Group" shows the groups on the map.
