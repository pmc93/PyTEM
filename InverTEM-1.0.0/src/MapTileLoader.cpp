#include "MapTileLoader.h"

#include <QColor>
#include <QDir>
#include <QNetworkAccessManager>
#include <QNetworkDiskCache>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPainter>
#include <QStandardPaths>

#include <algorithm>
#include <cmath>

namespace {

constexpr int tilePixels = 256;
constexpr int tileGridSize = 7;
constexpr int totalTiles = tileGridSize * tileGridSize;
constexpr int maximumConcurrentRequests = 2;
constexpr double maximumMercatorLatitude = 85.05112878;

double longitudeToTileX(double longitude, int zoom)
{
    const double scale = std::ldexp(1.0, zoom);
    return (longitude + 180.0) / 360.0 * scale;
}

double latitudeToTileY(double latitude, int zoom)
{
    const double bounded = std::clamp(latitude,
        -maximumMercatorLatitude, maximumMercatorLatitude);
    const double radians = bounded * 3.14159265358979323846 / 180.0;
    const double scale = std::ldexp(1.0, zoom);
    return (1.0 - std::asinh(std::tan(radians)) / 3.14159265358979323846)
        * 0.5 * scale;
}

double tileXToLongitude(double x, int zoom)
{
    return x / std::ldexp(1.0, zoom) * 360.0 - 180.0;
}

double tileYToLatitude(double y, int zoom)
{
    const double n = 3.14159265358979323846
        - 2.0 * 3.14159265358979323846 * y / std::ldexp(1.0, zoom);
    return 180.0 / 3.14159265358979323846 * std::atan(std::sinh(n));
}

} // namespace

MapTileLoader::MapTileLoader(QObject *parent) : QObject(parent)
{
    m_manager = new QNetworkAccessManager(this);
    auto *cache = new QNetworkDiskCache(m_manager);
    QString cacheRoot = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (cacheRoot.isEmpty())
        cacheRoot = QDir::tempPath() + "/invertem-cache";
    cache->setCacheDirectory(cacheRoot + "/map_tiles");
    cache->setMaximumCacheSize(200LL * 1024LL * 1024LL);
    m_manager->setCache(cache);
}

void MapTileLoader::cancel()
{
    ++m_generation;
    m_pending.clear();
    const auto replies = m_activeReplies;
    m_activeReplies.clear();
    for (QNetworkReply *reply : replies)
        reply->abort();
}

void MapTileLoader::load(Style style, double minimumLongitude,
                         double maximumLongitude, double minimumLatitude,
                         double maximumLatitude)
{
    cancel();
    minimumLongitude = std::clamp(minimumLongitude, -180.0, 180.0);
    maximumLongitude = std::clamp(maximumLongitude, -180.0, 180.0);
    minimumLatitude = std::clamp(minimumLatitude,
        -maximumMercatorLatitude, maximumMercatorLatitude);
    maximumLatitude = std::clamp(maximumLatitude,
        -maximumMercatorLatitude, maximumMercatorLatitude);
    if (minimumLongitude > maximumLongitude)
        std::swap(minimumLongitude, maximumLongitude);
    if (minimumLatitude > maximumLatitude)
        std::swap(minimumLatitude, maximumLatitude);

    int zoom = 3;
    for (int candidate = 18; candidate >= 3; --candidate) {
        const double xSpan = longitudeToTileX(maximumLongitude, candidate)
            - longitudeToTileX(minimumLongitude, candidate);
        const double ySpan = latitudeToTileY(minimumLatitude, candidate)
            - latitudeToTileY(maximumLatitude, candidate);
        if (xSpan <= static_cast<double>(tileGridSize - 2)
            && ySpan <= static_cast<double>(tileGridSize - 2)) {
            zoom = candidate;
            break;
        }
    }

    const int tileCountAtZoom = 1 << zoom;
    const double centerX = 0.5 * (longitudeToTileX(minimumLongitude, zoom)
                                  + longitudeToTileX(maximumLongitude, zoom));
    const double centerY = 0.5 * (latitudeToTileY(minimumLatitude, zoom)
                                  + latitudeToTileY(maximumLatitude, zoom));
    const int halfGrid = tileGridSize / 2;
    const int startX = std::clamp(static_cast<int>(std::floor(centerX)) - halfGrid,
                                  0, tileCountAtZoom - tileGridSize);
    const int startY = std::clamp(static_cast<int>(std::floor(centerY)) - halfGrid,
                                  0, tileCountAtZoom - tileGridSize);

    const double west = tileXToLongitude(startX, zoom);
    const double east = tileXToLongitude(startX + tileGridSize, zoom);
    const double north = tileYToLatitude(startY, zoom);
    const double south = tileYToLatitude(startY + tileGridSize, zoom);
    m_bounds = QRectF(west, south, east - west, north - south);
    m_mosaic = QImage(tilePixels * tileGridSize, tilePixels * tileGridSize,
                      QImage::Format_RGB32);
    m_mosaic.fill(QColor(232, 235, 238));
    m_completed = 0;
    m_failed = 0;
    const int generation = m_generation;

    if (style == Style::OpenStreetMap)
        m_attribution = QString::fromUtf8("© OpenStreetMap contributors");
    else
        m_attribution = "Source: Esri, Maxar, Earthstar Geographics, GIS User Community";

    for (int row = 0; row < tileGridSize; ++row) {
        for (int column = 0; column < tileGridSize; ++column) {
            const int tileX = startX + column;
            const int tileY = startY + row;
            QString url;
            if (style == Style::OpenStreetMap) {
                url = QString("https://tile.openstreetmap.org/%1/%2/%3.png")
                    .arg(zoom).arg(tileX).arg(tileY);
            } else {
                url = QString("https://server.arcgisonline.com/ArcGIS/rest/services/"
                              "World_Imagery/MapServer/tile/%1/%2/%3")
                    .arg(zoom).arg(tileY).arg(tileX);
            }
            m_pending.enqueue({QUrl(url), column, row, generation});
        }
    }
    emit progress(0, totalTiles);
    startNextRequests();
}

void MapTileLoader::startNextRequests()
{
    while (m_activeReplies.size() < maximumConcurrentRequests
           && !m_pending.isEmpty()) {
        const TileTask task = m_pending.dequeue();
        QNetworkRequest request(task.url);
        request.setRawHeader("User-Agent", "InverTEM/1.0.0");
        request.setRawHeader("Accept", "image/png,image/jpeg,image/*");
        request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                             QNetworkRequest::PreferCache);
        QNetworkReply *reply = m_manager->get(request);
        m_activeReplies.insert(reply);
        connect(reply, &QNetworkReply::finished, this,
                [this, reply, task]() { finishRequest(reply, task); });
    }
}

void MapTileLoader::finishRequest(QNetworkReply *reply, const TileTask &task)
{
    m_activeReplies.remove(reply);
    if (task.generation != m_generation) {
        reply->deleteLater();
        return;
    }

    QImage tile;
    const bool ok = reply->error() == QNetworkReply::NoError
        && tile.loadFromData(reply->readAll());
    if (ok) {
        QPainter painter(&m_mosaic);
        painter.drawImage(QRect(task.column * tilePixels, task.row * tilePixels,
                                tilePixels, tilePixels), tile);
    } else {
        ++m_failed;
    }
    ++m_completed;
    emit progress(m_completed, totalTiles);
    reply->deleteLater();

    if (m_completed == totalTiles) {
        if (m_failed == totalTiles)
            emit failed("No background-map tiles could be downloaded. Check the internet connection.");
        else
            emit loaded(m_mosaic, m_bounds, m_attribution, m_failed);
        return;
    }
    startNextRequests();
}
