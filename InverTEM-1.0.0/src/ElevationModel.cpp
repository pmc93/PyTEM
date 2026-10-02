#include "ElevationModel.h"

#include <QByteArray>
#include <QFile>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace pytem {
namespace {

// TIFF LZW: MSB-first codes of 9-12 bits, widened one code early.
std::vector<unsigned char> lzwDecode(const unsigned char *data, std::size_t size, std::size_t expected)
{
    std::vector<int> prefix(4096, -1);
    std::vector<unsigned char> suffix(4096), stack, out;
    for (int i = 0; i < 256; ++i)
        suffix[i] = static_cast<unsigned char>(i);
    out.reserve(expected);
    int bits = 9, next = 258, old = -1;
    unsigned char first = 0;
    std::size_t bit = 0;
    while (bit + bits <= size * 8) {
        int code = 0;
        for (int i = 0; i < bits; ++i, ++bit)
            code = (code << 1) | ((data[bit >> 3] >> (7 - (bit & 7))) & 1);
        if (code == 257)
            break;
        if (code == 256) {
            bits = 9, next = 258, old = -1;
            continue;
        }
        if (code > next || (old < 0 && code >= 256))
            throw std::runtime_error("corrupt LZW data");
        stack.clear();
        int c = code;
        if (code == next) { // the string being defined: previous + its first byte
            stack.push_back(first);
            c = old;
        }
        for (; c >= 0; c = prefix[c])
            stack.push_back(suffix[c]);
        first = stack.back();
        out.insert(out.end(), stack.rbegin(), stack.rend());
        if (old >= 0 && next < 4096) {
            prefix[next] = old;
            suffix[next] = first;
            ++next;
        }
        old = code;
        if (next >= (1 << bits) - 1 && bits < 12)
            ++bits;
    }
    return out;
}

ElevationGrid readGeoTiff(const std::string &path)
{
    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly))
        throw std::runtime_error("cannot open " + path);
    const QByteArray bytes = file.readAll();
    const auto *d = reinterpret_cast<const unsigned char *>(bytes.constData());
    const std::size_t n = static_cast<std::size_t>(bytes.size());
    if (n < 8 || !((d[0] == 'I' && d[1] == 'I') || (d[0] == 'M' && d[1] == 'M')))
        throw std::runtime_error("not a TIFF file");
    const bool little = d[0] == 'I';
    auto at = [&](std::size_t offset, int size) {
        if (offset + size > n)
            throw std::runtime_error("truncated TIFF file");
        std::uint64_t value = 0;
        for (int i = 0; i < size; ++i)
            value |= std::uint64_t(d[offset + (little ? i : size - 1 - i)]) << (8 * i);
        return value;
    };
    if (at(2, 2) != 42)
        throw std::runtime_error("BigTIFF is not supported");

    // Tags of the first (full resolution) image, as numbers.
    const std::size_t ifd = at(4, 4);
    std::map<int, std::vector<double>> tags;
    std::string noData;
    for (std::size_t e = ifd + 2, count = at(ifd, 2); count-- > 0; e += 12) {
        const int tag = static_cast<int>(at(e, 2)), type = static_cast<int>(at(e + 2, 2));
        const std::size_t items = at(e + 4, 4);
        static const int sizes[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};
        const int size = type > 0 && type <= 12 ? sizes[type] : 1;
        const std::size_t start = items * size <= 4 ? e + 8 : at(e + 8, 4);
        if (type == 2) {
            if (tag == 42113) // GDAL_NODATA
                noData.assign(reinterpret_cast<const char *>(d + start), std::min(items, n - start));
            continue;
        }
        auto &values = tags[tag];
        for (std::size_t i = 0; i < items; ++i) {
            const std::size_t o = start + i * size;
            std::uint64_t raw = at(o, type == 5 || type == 10 ? 4 : size);
            double value;
            if (type == 11) { float f; const auto r = static_cast<std::uint32_t>(raw); std::memcpy(&f, &r, 4); value = f; }
            else if (type == 12) std::memcpy(&value, &raw, 8);
            else if (type == 5 || type == 10) value = double(raw) / double(std::max<std::uint64_t>(1, at(o + 4, 4)));
            else if (type == 6 || type == 8 || type == 9) value = static_cast<double>(static_cast<std::int64_t>(raw << (64 - 8 * size)) >> (64 - 8 * size));
            else value = static_cast<double>(raw);
            values.push_back(value);
        }
    }
    auto tag = [&](int id, double fallback) { return tags.count(id) && !tags[id].empty() ? tags[id][0] : fallback; };

    ElevationGrid grid;
    grid.width = static_cast<int>(tag(256, 0));
    grid.height = static_cast<int>(tag(257, 0));
    const int bitsPerSample = static_cast<int>(tag(258, 8)), compression = static_cast<int>(tag(259, 1));
    const int samplesPerPixel = static_cast<int>(tag(277, 1)), predictor = static_cast<int>(tag(317, 1));
    const int sampleFormat = static_cast<int>(tag(339, 1)), bytesPerSample = bitsPerSample / 8;
    if (grid.width <= 0 || grid.height <= 0 || !tags.count(33550) || !tags.count(33922))
        throw std::runtime_error("not a georeferenced single-band GeoTIFF");
    if (compression != 1 && compression != 5 && compression != 8 && compression != 32946)
        throw std::runtime_error("unsupported TIFF compression " + std::to_string(compression));
    if (bytesPerSample != 2 && bytesPerSample != 4 && bytesPerSample != 8)
        throw std::runtime_error("unsupported TIFF sample size");

    const bool tiled = tags.count(324) > 0;
    const int chunkWidth = tiled ? static_cast<int>(tag(322, 0)) : grid.width;
    const int chunkHeight = tiled ? static_cast<int>(tag(323, 0)) : static_cast<int>(tag(278, grid.height));
    const auto &offsets = tags[tiled ? 324 : 273], &counts = tags[tiled ? 325 : 279];
    const int across = (grid.width + chunkWidth - 1) / chunkWidth;
    const std::size_t pixelBytes = static_cast<std::size_t>(bytesPerSample) * samplesPerPixel;
    grid.z.assign(static_cast<std::size_t>(grid.width) * grid.height, std::numeric_limits<float>::quiet_NaN());
    const double noDataValue = noData.empty() ? std::numeric_limits<double>::quiet_NaN() : std::atof(noData.c_str());
    for (std::size_t chunk = 0; chunk < offsets.size() && chunk < counts.size(); ++chunk) {
        const std::size_t expected = static_cast<std::size_t>(chunkWidth) * chunkHeight * pixelBytes;
        const auto *source = d + static_cast<std::size_t>(offsets[chunk]);
        const auto sourceSize = static_cast<std::size_t>(counts[chunk]);
        if (static_cast<std::size_t>(offsets[chunk]) + sourceSize > n)
            throw std::runtime_error("truncated TIFF file");
        std::vector<unsigned char> raw;
        if (compression == 5) {
            raw = lzwDecode(source, sourceSize, expected);
        } else if (compression == 8 || compression == 32946) {
            QByteArray zlib(4, 0); // qUncompress expects the expected size in front
            for (int i = 0; i < 4; ++i)
                zlib[i] = static_cast<char>((expected >> (24 - 8 * i)) & 0xff);
            zlib.append(reinterpret_cast<const char *>(source), static_cast<int>(sourceSize));
            const QByteArray out = qUncompress(zlib);
            raw.assign(out.begin(), out.end());
        } else {
            raw.assign(source, source + sourceSize);
        }
        raw.resize(expected, 0);
        const int rows = static_cast<int>(expected / (chunkWidth * pixelBytes));
        const std::size_t rowBytes = static_cast<std::size_t>(chunkWidth) * pixelBytes;
        bool native = little; // raw is in file byte order unless reordered below (hosts are little-endian)
        for (int r = 0; r < rows; ++r) {
            unsigned char *row = raw.data() + r * rowBytes;
            if (predictor == 3) { // byte differences, then big-endian byte planes -> host order
                for (std::size_t i = 1; i < rowBytes; ++i)
                    row[i] = static_cast<unsigned char>(row[i] + row[i - 1]);
                const std::vector<unsigned char> planes(row, row + rowBytes);
                for (int i = 0; i < chunkWidth * samplesPerPixel; ++i)
                    for (int b = 0; b < bytesPerSample; ++b)
                        row[i * bytesPerSample + b] = planes[(bytesPerSample - 1 - b) * chunkWidth * samplesPerPixel + i];
                native = true;
            }
        }
        const int chunkX = static_cast<int>(chunk % across) * chunkWidth;
        const int chunkY = tiled ? static_cast<int>(chunk / across) * chunkHeight : static_cast<int>(chunk) * chunkHeight;
        for (int r = 0; r < rows && chunkY + r < grid.height; ++r) {
            std::int64_t previous = 0;
            for (int c = 0; c < chunkWidth; ++c) {
                const unsigned char *p = raw.data() + r * rowBytes + c * pixelBytes;
                std::uint64_t bits = 0;
                for (int b = 0; b < bytesPerSample; ++b)
                    bits |= std::uint64_t(p[native ? b : bytesPerSample - 1 - b]) << (8 * b);
                double value;
                if (sampleFormat == 3) {
                    if (bytesPerSample == 4) { float f; const auto v = static_cast<std::uint32_t>(bits); std::memcpy(&f, &v, 4); value = f; }
                    else std::memcpy(&value, &bits, 8);
                } else {
                    std::int64_t integer = sampleFormat == 2
                        ? static_cast<std::int64_t>(bits << (64 - 8 * bytesPerSample)) >> (64 - 8 * bytesPerSample)
                        : static_cast<std::int64_t>(bits);
                    if (predictor == 2) { // horizontal differencing, wrapping at the sample size
                        integer += previous;
                        const int shift = 64 - 8 * bytesPerSample;
                        integer = sampleFormat == 2 ? (integer << shift) >> shift
                                                    : static_cast<std::int64_t>(static_cast<std::uint64_t>(integer << shift) >> shift);
                        previous = integer;
                    }
                    value = static_cast<double>(integer);
                }
                if (chunkX + c < grid.width && !(value == noDataValue) && value > -1e30 && std::isfinite(value))
                    grid.z[static_cast<std::size_t>(chunkY + r) * grid.width + chunkX + c] = static_cast<float>(value);
            }
        }
    }

    // Georeferencing: pixel scale and a tie point at pixel (i, j); pixel
    // corners unless GTRasterTypeGeoKey says the tie point is a pixel centre.
    const auto &scale = tags[33550], &tie = tags[33922];
    bool pixelIsPoint = false;
    if (tags.count(34735))
        for (std::size_t k = 4; k + 3 < tags[34735].size(); k += 4) {
            const auto &keys = tags[34735];
            if (keys[k] == 1025 && keys[k + 1] == 0)
                pixelIsPoint = keys[k + 3] == 2;
            if ((keys[k] == 3072 || keys[k] == 2048) && keys[k + 1] == 0) // projected or geographic CRS
                grid.epsg = static_cast<int>(keys[k + 3]);
        }
    grid.dx = scale[0];
    grid.dy = scale[1];
    const double half = pixelIsPoint ? 0.0 : 0.5;
    grid.x0 = tie[3] + (half - tie[0]) * grid.dx;
    grid.y0 = tie[4] - (half - tie[1]) * grid.dy;
    return grid;
}

// "x y z" rows (spaces, commas or semicolons) on a regular grid.
ElevationGrid readXyzGrid(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open " + path);
    std::vector<double> xs, ys, zs;
    for (std::string line; std::getline(file, line);) {
        std::replace_if(line.begin(), line.end(), [](char c) { return c == ',' || c == ';'; }, ' ');
        std::istringstream row(line);
        double x, y, z;
        if (row >> x >> y >> z)
            xs.push_back(x), ys.push_back(y), zs.push_back(z);
    }
    if (xs.size() < 4)
        throw std::runtime_error("no x y z rows in " + path);
    auto step = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        double smallest = std::numeric_limits<double>::infinity();
        for (std::size_t i = 1; i < values.size(); ++i)
            if (values[i] - values[i - 1] > 1e-6)
                smallest = std::min(smallest, values[i] - values[i - 1]);
        return std::make_pair(values.front(), std::isfinite(smallest) ? smallest : 1.0);
    };
    const auto [xMin, dx] = step(xs);
    const auto [yMin, dy] = step(ys);
    ElevationGrid grid;
    grid.dx = dx, grid.dy = dy, grid.x0 = xMin;
    grid.y0 = *std::max_element(ys.begin(), ys.end());
    grid.width = static_cast<int>(std::lround((*std::max_element(xs.begin(), xs.end()) - xMin) / dx)) + 1;
    grid.height = static_cast<int>(std::lround((grid.y0 - yMin) / dy)) + 1;
    grid.z.assign(static_cast<std::size_t>(grid.width) * grid.height, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < xs.size(); ++i)
        grid.z[static_cast<std::size_t>(std::lround((grid.y0 - ys[i]) / dy)) * grid.width
               + std::lround((xs[i] - xMin) / dx)] = static_cast<float>(zs[i]);
    return grid;
}

} // namespace

ElevationGrid readElevationGrid(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    char magic[2] = {};
    file.read(magic, 2);
    return (magic[0] == 'I' && magic[1] == 'I') || (magic[0] == 'M' && magic[1] == 'M')
        ? readGeoTiff(path) : readXyzGrid(path);
}

double sampleElevation(const std::vector<ElevationGrid> &grids, double x, double y)
{
    for (const auto &g : grids) {
        const double fc = (x - g.x0) / g.dx, fr = (g.y0 - y) / g.dy;
        if (fc < -0.5 || fr < -0.5 || fc > g.width - 0.5 || fr > g.height - 0.5)
            continue;
        const int c0 = std::clamp(static_cast<int>(std::floor(fc)), 0, std::max(0, g.width - 2));
        const int r0 = std::clamp(static_cast<int>(std::floor(fr)), 0, std::max(0, g.height - 2));
        const int c1 = std::min(c0 + 1, g.width - 1), r1 = std::min(r0 + 1, g.height - 1);
        const double u = std::clamp(fc - c0, 0.0, 1.0), v = std::clamp(fr - r0, 0.0, 1.0);
        auto z = [&](int c, int r) { return static_cast<double>(g.z[static_cast<std::size_t>(r) * g.width + c]); };
        const double value = (1 - v) * ((1 - u) * z(c0, r0) + u * z(c1, r0)) + v * ((1 - u) * z(c0, r1) + u * z(c1, r1));
        if (std::isfinite(value))
            return value;
        const double nearest = z(u < 0.5 ? c0 : c1, v < 0.5 ? r0 : r1); // next to a no-data pixel
        if (std::isfinite(nearest))
            return nearest;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace pytem
