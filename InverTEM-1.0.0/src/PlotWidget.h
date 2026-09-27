#pragma once

#include <QColor>
#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QVector>
#include <QWidget>
#include <limits>

class PlotWidget final : public QWidget
{
    Q_OBJECT

public:
    struct Curve {
        QString name;
        QVector<QPointF> points;
        QColor color;
        bool markers = false;
        bool stepped = false;
        bool selectable = false;
        QVector<int> pointIds;
        bool line = true;
        QVector<double> yErrors;
        bool dashed = false;
        bool filledMarkers = false;
    };

    explicit PlotWidget(QWidget *parent = nullptr);

    void setAxes(QString title, QString xLabel, QString yLabel,
                 bool logX, bool logY, bool invertY = false);
    void setCurves(QVector<Curve> curves);
    void setBackgroundImage(const QImage &image, const QRectF &dataBounds,
                            QString attribution = {});
    void clearBackgroundImage();
    void setEqualAspect(bool enabled, double xScale = 1.0);
    void setGeographicScaleBar(bool enabled);
    void setLegendBackground(bool enabled);
    void setTickLabelsVisible(bool visible);
    void setMarkerRadius(double radius);
    void setMinimumY(double value);
    void setScientificX(bool enabled);
    void setNiceLogXTicks(bool enabled);
    void setYTickIntervals(int intervals); // grid lines/labels on the y axis minus one (default 5)
    void setXRange(double minimum, double maximum);
    void clearXRange();
    void setColorScale(QString title, double minimum, double maximum);
    void clearColorScale();
    static QColor colorScaleColor(double value, double minimum,
                                  double maximum);
    void clear();

signals:
    void pointClicked(int pointId);
    void pointRightClicked(int pointId);
    void pointsRightDragged(const QVector<int> &pointIds, bool restore); // restore = Shift held

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    QPointF mapPoint(const QPointF &point, const QRectF &area,
                     double xMin, double xMax, double yMin, double yMax) const;
    double transformX(double value) const;
    double transformY(double value) const;

    QString m_title;
    QString m_xLabel;
    QString m_yLabel;
    QVector<Curve> m_curves;
    QImage m_backgroundImage;
    QRectF m_backgroundBounds;
    QString m_backgroundAttribution;
    struct HitPoint {
        QPointF position;
        int pointId = -1;
    };
    QVector<HitPoint> m_hitPoints;
    bool m_logX = true;
    bool m_logY = true;
    bool m_invertY = false;
    bool m_equalAspect = false;
    double m_xAspectScale = 1.0;
    bool m_geographicScaleBar = false;
    bool m_legendBackground = false;
    bool m_tickLabelsVisible = true;
    double m_markerRadius = 3.2;
    double m_minimumY = std::numeric_limits<double>::quiet_NaN();
    bool m_scientificX = false;
    bool m_niceLogXTicks = false;
    int m_yTickIntervals = 5;
    double m_xMinimum = std::numeric_limits<double>::quiet_NaN();
    double m_xMaximum = std::numeric_limits<double>::quiet_NaN();
    QString m_colorScaleTitle;
    double m_colorScaleMinimum = std::numeric_limits<double>::quiet_NaN();
    double m_colorScaleMaximum = std::numeric_limits<double>::quiet_NaN();
    bool m_rightDragging = false;
    QPointF m_dragStart;
    QPointF m_dragCurrent;
};
