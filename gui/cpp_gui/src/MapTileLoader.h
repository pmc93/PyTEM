#pragma once

#include <QImage>
#include <QObject>
#include <QQueue>
#include <QRectF>
#include <QSet>
#include <QString>
#include <QUrl>

class QNetworkAccessManager;
class QNetworkReply;

class MapTileLoader final : public QObject
{
    Q_OBJECT

public:
    enum class Style {
        OpenStreetMap,
        Satellite,
    };
    Q_ENUM(Style)

    explicit MapTileLoader(QObject *parent = nullptr);

    void load(Style style, double minimumLongitude, double maximumLongitude,
              double minimumLatitude, double maximumLatitude);
    void cancel();

signals:
    void progress(int completedTiles, int totalTiles);
    void loaded(const QImage &image, const QRectF &longitudeLatitudeBounds,
                const QString &attribution, int failedTiles);
    void failed(const QString &message);

private:
    struct TileTask {
        QUrl url;
        int column = 0;
        int row = 0;
        int generation = 0;
    };

    void startNextRequests();
    void finishRequest(QNetworkReply *reply, const TileTask &task);

    QNetworkAccessManager *m_manager = nullptr;
    QQueue<TileTask> m_pending;
    QSet<QNetworkReply *> m_activeReplies;
    QImage m_mosaic;
    QRectF m_bounds;
    QString m_attribution;
    int m_generation = 0;
    int m_completed = 0;
    int m_failed = 0;
};
