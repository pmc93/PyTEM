#pragma once

#include <string>
#include <vector>

namespace pytem {

// A terrain model grid: z[row * width + col] is the height at the centre of
// pixel (col, row), at x0 + col * dx, y0 - row * dy. NaN where there is no data.
struct ElevationGrid {
    double x0 = 0.0, y0 = 0.0, dx = 1.0, dy = 1.0;
    int width = 0, height = 0;
    int epsg = 0; // 0 when the file does not say
    std::vector<float> z;
};

// GeoTIFF (classic TIFF; strips or tiles; uncompressed, LZW or Deflate;
// predictors 1-3; 16/32-bit integers or 32/64-bit floats), e.g. the Lower
// Saxony DGM1 or the Danish DHM tiles, or "x y z" text on a regular grid.
ElevationGrid readElevationGrid(const std::string &path);

// Bilinear height at (x, y) from the first grid that covers it; NaN outside.
double sampleElevation(const std::vector<ElevationGrid> &grids, double x, double y);

} // namespace pytem
